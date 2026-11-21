// faradice GPU transition walk (f32, coordinates relative to a scene origin).
//
// This is a port of `faradice_frw::walk::Walker::run` and of the uniform-grid
// queries in `faradice_geom::grid`. Keep the two in sync.

const RUNNING: u32 = 0xFFFFFFFEu;
const ABORTED: u32 = 0xFFFFFFFFu;
const SINK_GROUND_PLANE: u32 = 0xFFFFFF00u;
const SINK_FACE0: u32 = 0xFFFFFF10u; // + Face::index()
const SINK_INFINITY: u32 = 0xFFFFFF20u;
const BIG: f32 = 3.0e38;

struct Params {
    grid_origin: vec4<f32>, // xyz origin, w cell size
    grid_dims: vec4<u32>,   // xyz dims, w = 1 if the near-cell cache is present
    bounds_lo: vec4<f32>,   // conductor bounds, w = eps_stop
    bounds_hi: vec4<f32>,   // w = far_step
    dom_lo: vec4<f32>,      // domain box (if any), w = 1 if a box domain
    dom_hi: vec4<f32>,      // w = substrate z
    sphere: vec4<f32>,      // open boundary: centre xyz, radius w (0 = none)
    flags: vec4<u32>,       // x has_substrate, y reflect bits, z ground-face bits, w green resolution m
    ctl: vec4<u32>,         // x n_walks, y steps per dispatch, z max_steps, w = 1 on the first round
    offs0: vec4<u32>,       // offsets in `idx`: cell_start, items, clearance, near_start
    offs1: vec4<u32>,       // near_items, alias_prob (f32 bits), alias_idx, box_net
}

struct Walk {
    pos: vec3<f32>,
    sign: f32,
    rng: vec4<u32>,
    steps: u32,
    status: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> boxes: array<vec4<f32>>; // 2 per box: (lo, 0), (hi, 0)
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> walks: array<Walk>;
@group(0) @binding(4) var<storage, read_write> running: atomic<u32>;
// Launches: 5 u32 per walk (x, y, z as f32 bits; key low, key high).
@group(0) @binding(5) var<storage, read> launches: array<u32>;
// Outcomes: (status, steps | sign bit << 31) per walk.
@group(0) @binding(6) var<storage, read_write> outcomes: array<vec2<u32>>;

fn fmix32(h_in: u32) -> u32 {
    var h = h_in;
    h ^= h >> 16u;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13u;
    h *= 0xC2B2AE35u;
    h ^= h >> 16u;
    return h;
}

// ---- xoshiro128++ (Blackman & Vigna, public domain) ----
var<private> S: vec4<u32>;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next_u32() -> u32 {
    let result = rotl(S.x + S.w, 7u) + S.x;
    let t = S.y << 9u;
    S.z ^= S.x;
    S.w ^= S.y;
    S.y ^= S.z;
    S.x ^= S.w;
    S.z ^= t;
    S.w = rotl(S.w, 11u);
    return result;
}

// Uniform in [0, 1) with 24 random bits.
fn next_f32() -> f32 {
    return f32(next_u32() >> 8u) * (1.0 / 16777216.0);
}

// Uniform integer in 0..n for n < 2^16 (multiply-shift, bias < n / 2^32).
fn below(n: u32) -> u32 {
    let x = next_u32();
    let hi = (x >> 16u) * n;
    let lo = ((x & 0xFFFFu) * n) >> 16u;
    return (hi + lo) >> 16u;
}

// ---- geometry ----
fn box_lo(i: u32) -> vec4<f32> { return boxes[2u * i]; }
fn box_hi(i: u32) -> vec3<f32> { return boxes[2u * i + 1u].xyz; }

fn linf_to_box(p: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    let d = max(lo - p, p - hi);
    return max(max(d.x, d.y), max(d.z, 0.0));
}

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    let f = floor((p - P.grid_origin.xyz) / P.grid_origin.w);
    return vec3<i32>(clamp(f, vec3<f32>(-1.0e9), vec3<f32>(1.0e9)));
}

fn inside_grid(c: vec3<i32>) -> bool {
    let n = vec3<i32>(P.grid_dims.xyz);
    return all(c >= vec3<i32>(0)) && all(c < n);
}

fn linear(c: vec3<i32>) -> u32 {
    let n = P.grid_dims.xyz;
    return (u32(c.z) * n.y + u32(c.y)) * n.x + u32(c.x);
}

struct Best {
    d: f32,
    id: u32,
}

fn consider(p: vec3<f32>, max_r: f32, i: u32, best: ptr<function, Best>) {
    let lo = box_lo(i);
    let d = linf_to_box(p, lo.xyz, box_hi(i));
    if d > max_r {
        return;
    }
    if d < (*best).d || (d == (*best).d && i < (*best).id) {
        (*best).d = d;
        (*best).id = i;
    }
}

fn visit_cell(p: vec3<f32>, max_r: f32, lin: u32, best: ptr<function, Best>) {
    let s = idx[P.offs0.x + lin];
    let e = idx[P.offs0.x + lin + 1u];
    for (var j = s; j < e; j++) {
        consider(p, max_r, idx[P.offs0.y + j], best);
    }
}

fn visit_ring(p: vec3<f32>, max_r: f32, c: vec3<i32>, k: i32, best: ptr<function, Best>) {
    let n = vec3<i32>(P.grid_dims.xyz);
    let lo = max(c - vec3<i32>(k), vec3<i32>(0));
    let hi = min(c + vec3<i32>(k), n - vec3<i32>(1));
    for (var z = lo.z; z <= hi.z; z++) {
        let z_on = abs(z - c.z) == k;
        for (var y = lo.y; y <= hi.y; y++) {
            let y_on = abs(y - c.y) == k;
            if z_on || y_on {
                for (var x = lo.x; x <= hi.x; x++) {
                    visit_cell(p, max_r, linear(vec3<i32>(x, y, z)), best);
                }
            } else {
                let x0 = c.x - k;
                if x0 >= 0 && x0 < n.x {
                    visit_cell(p, max_r, linear(vec3<i32>(x0, y, z)), best);
                }
                let x1 = c.x + k;
                if k > 0 && x1 >= 0 && x1 < n.x {
                    visit_cell(p, max_r, linear(vec3<i32>(x1, y, z)), best);
                }
            }
        }
    }
}

// Conservative L∞ distance from p to the outside of the (2k+1)^3 block around c.
fn block_reach(p: vec3<f32>, c: vec3<i32>, k: i32) -> f32 {
    let h = P.grid_origin.w;
    let o = P.grid_origin.xyz;
    let lo = o + vec3<f32>(c - vec3<i32>(k)) * h;
    let hi = o + vec3<f32>(c + vec3<i32>(k + 1)) * h;
    let r = min(p - lo, hi - p);
    let reach = min(min(r.x, r.y), r.z);
    let scale = max(f32(k + 1) * h, max(max(abs(p.x), abs(p.y)), abs(p.z)));
    let g = 4.0e-7 * scale; // a few f32 ulps
    let fl = max(f32(k) * h - g, 0.0);
    return max(reach - g, min(fl, max(reach, 0.0)));
}

fn nearest(p: vec3<f32>, max_r: f32) -> Best {
    var best = Best(BIG, 0xFFFFFFFFu);
    let c = cell_of(p);
    let inside = inside_grid(c);
    if inside && P.grid_dims.w == 1u {
        let lin = linear(c);
        let s = idx[P.offs0.w + lin];
        let e = idx[P.offs0.w + lin + 1u];
        if s < e {
            for (var j = s; j < e; j++) {
                consider(p, max_r, idx[P.offs1.x + j], &best);
            }
            return best;
        }
    }
    let n = vec3<i32>(P.grid_dims.xyz);
    let a = max(max(-c, c - (n - vec3<i32>(1))), vec3<i32>(0));
    var k_start = max(max(a.x, a.y), a.z);
    let b = max(max(c, (n - vec3<i32>(1)) - c), vec3<i32>(0));
    let k_end = max(max(b.x, b.y), b.z);
    if inside {
        k_start = max(k_start, i32(idx[P.offs0.z + linear(c)]));
    }
    var k = k_start;
    loop {
        let lower = f32(max(k - 1, 0)) * P.grid_origin.w;
        if k > k_start && lower >= max_r {
            break;
        }
        visit_ring(p, max_r, c, k, &best);
        let reach = block_reach(p, c, k);
        if best.d <= reach || k >= k_end || reach >= max_r {
            break;
        }
        k += 1;
    }
    return best;
}

fn clearance_lower_bound(p: vec3<f32>) -> f32 {
    let c = cell_of(p);
    if inside_grid(c) {
        let m = idx[P.offs0.z + linear(c)];
        return f32(max(m, 1u) - 1u) * P.grid_origin.w;
    }
    return linf_to_box(p, P.bounds_lo.xyz, P.bounds_hi.xyz);
}

struct Clr {
    h: f32,
    sink: u32,
}

fn dirichlet_distance(p: vec3<f32>) -> Clr {
    var best = Clr(BIG, SINK_INFINITY);
    if P.flags.x == 1u {
        best = Clr(p.z - P.dom_hi.w, SINK_GROUND_PLANE);
    }
    let gf = P.flags.z;
    for (var f = 0u; f < 6u; f++) {
        if (gf & (1u << f)) != 0u {
            let ax = f >> 1u;
            var coord = P.dom_lo[ax];
            if (f & 1u) == 1u {
                coord = P.dom_hi[ax];
            }
            let d = abs(p[ax] - coord);
            if d < best.h {
                best = Clr(d, SINK_FACE0 + f);
            }
        }
    }
    return best;
}

fn clearance(p: vec3<f32>) -> Clr {
    let bnd = dirichlet_distance(p);
    let lb = clearance_lower_bound(p);
    if lb >= bnd.h {
        return bnd;
    }
    if lb >= P.bounds_hi.w && lb >= P.bounds_lo.w {
        return Clr(lb, bnd.sink);
    }
    let nb = nearest(p, bnd.h);
    if nb.id != 0xFFFFFFFFu && nb.d < bnd.h {
        return Clr(nb.d, idx[P.offs1.w + nb.id]);
    }
    return bnd;
}

fn fold(p_in: vec3<f32>) -> vec3<f32> {
    var p = p_in;
    if P.dom_lo.w != 1.0 {
        return p;
    }
    let bits = P.flags.y;
    for (var a = 0u; a < 3u; a++) {
        let rmin = (bits & (1u << (2u * a))) != 0u;
        let rmax = (bits & (1u << (2u * a + 1u))) != 0u;
        if !(rmin || rmax) {
            continue;
        }
        for (var it = 0; it < 64; it++) {
            if rmin && p[a] < P.dom_lo[a] {
                p[a] = 2.0 * P.dom_lo[a] - p[a];
            } else if rmax && p[a] > P.dom_hi[a] {
                p[a] = 2.0 * P.dom_hi[a] - p[a];
            } else {
                break;
            }
        }
    }
    return p;
}

fn sample_transition() -> vec3<f32> {
    let m = P.flags.w;
    let face = below(6u);
    var cell = below(m * m);
    let prob = bitcast<f32>(idx[P.offs1.y + cell]);
    if next_f32() >= prob {
        cell = idx[P.offs1.z + cell];
    }
    let mf = f32(m);
    let u = 2.0 * ((f32(cell % m) + next_f32()) / mf) - 1.0;
    let v = 2.0 * ((f32(cell / m) + next_f32()) / mf) - 1.0;
    var w = 1.0;
    if (face & 1u) == 0u {
        w = -1.0;
    }
    let ax = face >> 1u;
    if ax == 0u {
        return vec3<f32>(w, u, v);
    } else if ax == 1u {
        return vec3<f32>(u, w, v);
    }
    return vec3<f32>(u, v, w);
}

// One exterior harmonic-measure step of the escape sphere; returns
// (point, 1) on return to the sphere or (_, 0) on escape.
fn exterior_sphere_step(x: vec3<f32>, c: vec3<f32>, radius: f32, r: f32) -> vec4<f32> {
    if next_f32() >= radius / r {
        return vec4<f32>(0.0);
    }
    let a = 1.0 / (r - radius);
    let b = 1.0 / (r + radius);
    let s = 1.0 / (a - next_f32() * (a - b));
    let cos_t = clamp((r * r + radius * radius - s * s) / (2.0 * r * radius), -1.0, 1.0);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let phi = 6.283185307179586 * next_f32();
    let e = (x - c) / r;
    var helper = vec3<f32>(0.0, 1.0, 0.0);
    if abs(e.x) < 0.9 {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t1 = normalize(cross(e, helper));
    let t2 = cross(e, t1);
    let y = c + radius * (cos_t * e + sin_t * (cos(phi) * t1 + sin(phi) * t2));
    return vec4<f32>(y, 1.0);
}

@compute @workgroup_size(64)
fn walk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= P.ctl.x {
        return;
    }
    var w: Walk;
    if P.ctl.w == 1u {
        let b = 5u * i;
        w.pos = vec3<f32>(bitcast<f32>(launches[b]), bitcast<f32>(launches[b + 1u]), bitcast<f32>(launches[b + 2u]));
        let lo = launches[b + 3u];
        let hi = launches[b + 4u];
        let s0 = fmix32(lo ^ 0x9E3779B9u);
        let s1 = fmix32(hi ^ (s0 + 0x7F4A7C15u));
        let s2 = fmix32(lo ^ (s1 + 0xF39CC060u));
        let s3 = fmix32(hi ^ (s2 + 0x5CEDC834u));
        w.rng = vec4<u32>(s0, s1, s2, s3 | 1u);
        w.sign = 1.0;
        w.steps = 0u;
        w.status = RUNNING;
    } else {
        w = walks[i];
        if w.status != RUNNING {
            return;
        }
    }
    S = w.rng;
    var p = w.pos;
    let eps_stop = P.bounds_lo.w;
    for (var it = 0u; it < P.ctl.y; it++) {
        if w.steps >= P.ctl.z {
            w.status = ABORTED;
            break;
        }
        if P.sphere.w > 0.0 {
            let c = P.sphere.xyz;
            let radius = P.sphere.w;
            let r = length(p - c);
            if r > 2.0 * radius {
                let y = exterior_sphere_step(p, c, radius, r);
                if y.w == 0.0 {
                    w.status = SINK_INFINITY;
                    break;
                }
                var q = y.xyz;
                if P.flags.x == 1u && q.z < P.dom_hi.w {
                    q.z = 2.0 * P.dom_hi.w - q.z;
                    w.sign = -w.sign;
                }
                p = q;
                w.steps += 1u;
                continue;
            }
        }
        let cl = clearance(p);
        if cl.h < eps_stop {
            w.status = cl.sink;
            break;
        }
        p = fold(p + cl.h * sample_transition());
        w.steps += 1u;
    }
    w.pos = p;
    w.rng = S;
    walks[i] = w;
    if w.status == RUNNING {
        atomicAdd(&running, 1u);
    } else {
        var sign_bit = 0u;
        if w.sign < 0.0 {
            sign_bit = 0x80000000u;
        }
        outcomes[i] = vec2<u32>(w.status, w.steps | sign_bit);
    }
}
