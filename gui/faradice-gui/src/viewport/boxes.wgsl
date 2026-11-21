// Instanced, depth-tested boxes with simple Lambert shading.

struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    // x: selected net (u32 bits), y: 1 if something is selected, z: target-only dimming
    selection: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) bmin: vec3<f32>,
    @location(3) net: u32,
    @location(4) bmax: vec3<f32>,
    @location(5) flags: u32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Stable, well-separated hue per net (golden-ratio hue walk).
fn net_color(net: u32) -> vec3<f32> {
    let h = fract(f32(net) * 0.61803398875 + 0.12);
    let k = vec3<f32>(5.0, 3.0, 1.0);
    let p = abs(fract(vec3<f32>(h) + k / 6.0) * 6.0 - 3.0);
    let rgb = clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return mix(vec3<f32>(0.5), rgb, 0.65);
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    let world = v.bmin + (v.bmax - v.bmin) * v.pos;
    var out: VertexOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    var base = net_color(v.net);
    let is_target = (v.flags & 1u) != 0u;
    if (u.selection.y == 1u) {
        if (v.net == u.selection.x) {
            base = vec3<f32>(1.0, 0.78, 0.25);
        } else {
            base = mix(base, vec3<f32>(0.45), 0.6);
        }
    } else if (!is_target && u.selection.z == 1u) {
        base = mix(base, vec3<f32>(0.45), 0.7);
    }
    let lambert = max(dot(normalize(v.normal), normalize(u.light_dir.xyz)), 0.0);
    out.color = base * (0.35 + 0.65 * lambert);
    return out;
}

@fragment
fn fs_main(v: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(v.color, 1.0);
}
