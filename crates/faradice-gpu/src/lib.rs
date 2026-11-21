//! GPU walk engine for faradice (wgpu compute, Vulkan on Linux).
//!
//! [`GpuEngine`] implements [`faradice_frw::WalkEngine`]: the solver keeps the
//! launch phase on the CPU in `f64` and hands the transition walks to the GPU.
//!
//! # Precision model
//!
//! Walks run in `f32` with coordinates relative to the centre of the
//! conductors' bounding box. Rounding then perturbs positions and distances by
//! about one ulp of the largest coordinate, which is harmless as long as the
//! stopping distance is much larger. [`GpuEngine::new`] refuses scenes where
//! `eps_stop < GpuOptions::min_stop_ulps` ulps of the scene scale (for large
//! layouts, use the CPU solver or a larger stopping distance).
//!
//! # Reproducibility
//!
//! Each walk draws from its own xoshiro128++ stream seeded from its launch
//! key, so a given device and driver produce the same results regardless of
//! how walks are grouped. Results agree with the CPU engine statistically, not
//! bit for bit (`f32` arithmetic, different random streams).

#![forbid(unsafe_code)]

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use faradice_core::{Face, NetId, Sink};
use faradice_frw::{Launch, Outcome, Scene, WalkEngine};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("walk.wgsl");
const RUNNING: u32 = 0xFFFF_FFFE;
const ABORTED: u32 = 0xFFFF_FFFF;
const SINK_GROUND_PLANE: u32 = 0xFFFF_FF00;
const SINK_FACE0: u32 = 0xFFFF_FF10;
const SINK_INFINITY: u32 = 0xFFFF_FF20;
const WORKGROUP: u32 = 64;
/// Largest number of walks in one dispatch (one-dimensional workgroup grid).
const MAX_DISPATCH_WALKS: usize = 65_535 * WORKGROUP as usize;

/// GPU engine options.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuOptions {
    /// Walks per solver call (whole batches are grouped up to this size).
    pub walks_per_call: usize,
    /// Transition steps per dispatch. Bounded dispatches keep each GPU job
    /// short (no driver watchdog resets) and let finished walks drop out.
    pub steps_per_dispatch: u32,
    /// Minimum stopping distance, in `f32` ulps of the scene scale.
    pub min_stop_ulps: f64,
    /// Prefer a discrete/integrated GPU over a software rasterizer.
    pub allow_software: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            walks_per_call: 1 << 20,
            steps_per_dispatch: 256,
            min_stop_ulps: 32.0,
            allow_software: false,
        }
    }
}

/// Errors creating or running the GPU engine.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuError {
    /// No suitable adapter (GPU) was found.
    NoAdapter(String),
    /// Device creation or execution failed.
    Device(String),
    /// The scene is too large for `f32` walks at this stopping distance.
    Precision(String),
    /// The scene exceeds device limits.
    TooLarge(String),
    /// The scene uses a feature the GPU walk does not implement.
    Unsupported(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter(m) => write!(f, "no GPU adapter: {m}"),
            Self::Device(m) => write!(f, "GPU device error: {m}"),
            Self::Precision(m) => write!(f, "insufficient f32 precision: {m}"),
            Self::TooLarge(m) => write!(f, "scene too large for the GPU: {m}"),
            Self::Unsupported(m) => write!(f, "not supported on the GPU: {m}"),
        }
    }
}

impl std::error::Error for GpuError {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    grid_origin: [f32; 4],
    grid_dims: [u32; 4],
    bounds_lo: [f32; 4],
    bounds_hi: [f32; 4],
    dom_lo: [f32; 4],
    dom_hi: [f32; 4],
    sphere: [f32; 4],
    flags: [u32; 4],
    ctl: [u32; 4],
    offs0: [u32; 4],
    offs1: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuWalk {
    pos: [f32; 3],
    sign: f32,
    rng: [u32; 4],
    steps: u32,
    status: u32,
    pad: [u32; 2],
}

/// The GPU walk engine.
#[derive(Debug)]
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    boxes: wgpu::Buffer,
    idx: wgpu::Buffer,
    params: Params,
    shift: [f64; 3],
    num_nets: u32,
    opts: GpuOptions,
    adapter: String,
    max_buffer: u64,
    bufs: Option<Buffers>,
}

fn to_f32(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)]
    let r = v as f32;
    r
}

fn rel(p: [f64; 3], shift: [f64; 3]) -> [f32; 3] {
    [
        to_f32(p[0] - shift[0]),
        to_f32(p[1] - shift[1]),
        to_f32(p[2] - shift[2]),
    ]
}

fn u32_len(n: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(n).map_err(|_| GpuError::TooLarge(format!("{what}: {n} entries")))
}

/// Vose alias table over `weights` (f32 probabilities, u32 aliases).
fn alias_table(weights: &[f64]) -> (Vec<f32>, Vec<u32>) {
    let n = weights.len();
    let total: f64 = weights.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let scale = n as f64 / total;
    let mut scaled: Vec<f64> = weights.iter().map(|w| w * scale).collect();
    let mut prob = vec![1.0_f32; n];
    #[allow(clippy::cast_possible_truncation)]
    let mut alias: Vec<u32> = (0..n).map(|i| i as u32).collect();
    let (mut small, mut large): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| scaled[i] < 1.0);
    while let (Some(s), Some(&l)) = (small.pop(), large.last()) {
        prob[s] = to_f32(scaled[s]);
        #[allow(clippy::cast_possible_truncation)]
        {
            alias[s] = l as u32;
        }
        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        if scaled[l] < 1.0 {
            large.pop();
            small.push(l);
        }
    }
    (prob, alias)
}

impl GpuEngine {
    /// Uploads `scene` to the first suitable GPU.
    ///
    /// # Errors
    ///
    /// See [`GpuError`].
    pub fn new(scene: &Scene<'_>, opts: GpuOptions) -> Result<Self, GpuError> {
        let geom = scene.geometry;
        let g = scene.grid.raw();
        let b = scene.boundary;
        let m = scene.green.resolution();
        if scene.medium.is_some() {
            return Err(GpuError::Unsupported(
                "stratified dielectric stacks (use the CPU engine)".into(),
            ));
        }
        if scene.boxes.is_some() {
            return Err(GpuError::Unsupported(
                "dielectric boxes and conformal shells (use the CPU engine)".into(),
            ));
        }
        if scene.max_steps >= 1 << 31 {
            return Err(GpuError::TooLarge("max_steps must be < 2^31".into()));
        }
        if m * m >= 1 << 16 {
            return Err(GpuError::TooLarge("green_resolution must be < 256".into()));
        }

        // Local frame centred on the conductors.
        let (lo, hi) = (g.bounds.min(), g.bounds.max());
        let shift: [f64; 3] = std::array::from_fn(|a| 0.5 * (lo[a] + hi[a]));
        let mut scale: f64 = (0..3).map(|a| (hi[a] - lo[a]).abs()).fold(0.0, f64::max);
        if let Some(d) = b.bounds() {
            for ((l, h), s) in d.min().iter().zip(d.max()).zip(shift) {
                scale = scale.max((l - s).abs()).max((h - s).abs());
            }
        }
        if let Some((c, r)) = b.sphere() {
            // Walks wander up to ~2r from the sphere centre before jumping.
            for (ca, s) in c.iter().zip(shift) {
                scale = scale.max((ca - s).abs() + 3.0 * r);
            }
        }
        let ulp = scale * f64::from(f32::EPSILON);
        if scene.eps_stop < opts.min_stop_ulps * ulp {
            return Err(GpuError::Precision(format!(
                "eps_stop = {:.3e} µm is below {} ulps ({:.3e} µm) of the scene scale {:.3e} µm",
                scene.eps_stop,
                opts.min_stop_ulps,
                opts.min_stop_ulps * ulp,
                scale
            )));
        }

        // Index buffer: cell_start, items, clearance, near_start, near_items,
        // alias_prob (f32 bits), alias_idx, box_net.
        let (prob, alias) = alias_table(scene.green.face_probabilities());
        let box_net: Vec<u32> = (0..geom.boxes().len())
            .map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                geom.box_net(faradice_core::BoxId::new(i as u32)).raw()
            })
            .collect();
        if g.clearance.is_empty() {
            return Err(GpuError::Device(
                "grid built without the clearance transform".into(),
            ));
        }
        let has_near = !g.near_start.is_empty();
        let mut idx: Vec<u32> = Vec::new();
        let mut push = |v: &[u32]| -> Result<u32, GpuError> {
            let off = u32_len(idx.len(), "index buffer")?;
            idx.extend_from_slice(v);
            // Never bind an empty range.
            idx.push(0);
            Ok(off)
        };
        let o_cell_start = push(g.cell_start)?;
        let o_items = push(g.items)?;
        let o_clear = push(g.clearance)?;
        let o_near_start = push(g.near_start)?;
        let o_near_items = push(g.near_items)?;
        let prob_bits: Vec<u32> = prob.iter().map(|p| p.to_bits()).collect();
        let o_prob = push(&prob_bits)?;
        let o_alias = push(&alias)?;
        let o_net = push(&box_net)?;

        let mut boxes: Vec<[f32; 4]> = Vec::with_capacity(2 * g.boxes.len());
        for bx in g.boxes {
            let (l, h) = (rel(bx.min(), shift), rel(bx.max(), shift));
            boxes.push([l[0], l[1], l[2], 0.0]);
            boxes.push([h[0], h[1], h[2], 0.0]);
        }

        let mut flags = [0u32; 4];
        let (mut dom_lo, mut dom_hi) = ([0.0_f32; 4], [0.0_f32; 4]);
        if let Some(d) = b.bounds() {
            let (l, h) = (rel(d.min(), shift), rel(d.max(), shift));
            dom_lo = [l[0], l[1], l[2], 1.0];
            dom_hi = [h[0], h[1], h[2], 0.0];
            for (a, (rmin, rmax)) in b.reflect().into_iter().enumerate() {
                flags[1] |= u32::from(rmin) << (2 * a) | u32::from(rmax) << (2 * a + 1);
            }
            for &(axis, coord, sink) in b.ground_faces() {
                let Sink::Boundary(face) = sink else {
                    return Err(GpuError::Device("unexpected ground face sink".into()));
                };
                debug_assert_eq!(face.axis.index(), axis);
                debug_assert!((d.face_coord(face) - coord).abs() <= f64::EPSILON * scale);
                flags[2] |= 1 << face.index();
            }
        }
        if let Some(z0) = b.substrate_z() {
            flags[0] = 1;
            dom_hi[3] = to_f32(z0 - shift[2]);
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            flags[3] = m as u32;
        }
        let sphere = b.sphere().map_or([0.0; 4], |(c, r)| {
            let c = rel(c, shift);
            [c[0], c[1], c[2], to_f32(r)]
        });
        let dims = [
            u32_len(g.dims[0], "grid")?,
            u32_len(g.dims[1], "grid")?,
            u32_len(g.dims[2], "grid")?,
            u32::from(has_near),
        ];
        let go = rel(g.origin, shift);
        let (bl, bh) = (rel(lo, shift), rel(hi, shift));
        let params = Params {
            grid_origin: [go[0], go[1], go[2], to_f32(g.cell)],
            grid_dims: dims,
            bounds_lo: [bl[0], bl[1], bl[2], to_f32(scene.eps_stop)],
            bounds_hi: [bh[0], bh[1], bh[2], to_f32(scene.far_step)],
            dom_lo,
            dom_hi,
            sphere,
            flags,
            ctl: [0, opts.steps_per_dispatch.max(1), scene.max_steps, 0],
            offs0: [o_cell_start, o_items, o_clear, o_near_start],
            offs1: [o_near_items, o_prob, o_alias, o_net],
        };

        let (device, queue, adapter, max_buffer) = Self::device(&opts)?;
        let need = (idx.len() * 4).max(boxes.len() * 16) as u64;
        if need > max_buffer {
            return Err(GpuError::TooLarge(format!(
                "{need} bytes exceed the storage binding limit {max_buffer}"
            )));
        }
        let boxes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boxes"),
            contents: bytemuck::cast_slice(&boxes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let idx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idx"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("walk.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("walk"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, true),
                storage(6, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("walk"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("walk"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("walk_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            boxes,
            idx,
            params,
            shift,
            num_nets: u32_len(geom.num_nets(), "nets")?,
            opts,
            adapter,
            max_buffer,
            bufs: None,
        })
    }

    fn device(opts: &GpuOptions) -> Result<(wgpu::Device, wgpu::Queue, String, u64), GpuError> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !opts.allow_software {
            return Err(GpuError::NoAdapter(format!(
                "only a software adapter is available ({})",
                info.name
            )));
        }
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("faradice"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| GpuError::Device(e.to_string()))?;
        let max = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        Ok((
            device,
            queue,
            format!("{} ({:?}, {})", info.name, info.device_type, info.driver),
            max,
        ))
    }

    /// Adapter description, e.g. `"Radeon 8060S Graphics (RADV …) (IntegratedGpu, radv)"`.
    #[must_use]
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    fn decode(&self, w: [u32; 2]) -> Result<Outcome, String> {
        let sink = match w[0] {
            ABORTED => None,
            SINK_GROUND_PLANE => Some(Sink::GroundPlane),
            SINK_INFINITY => Some(Sink::Infinity),
            s if (SINK_FACE0..SINK_FACE0 + 6).contains(&s) => {
                Some(Sink::Boundary(Face::ALL[(s - SINK_FACE0) as usize]))
            }
            s if s < self.num_nets => Some(Sink::Net(NetId::new(s))),
            RUNNING => return Err("walk still running after the step budget".into()),
            s => return Err(format!("invalid walk status {s:#x}")),
        };
        Ok(Outcome {
            sink,
            sign: if w[1] & 0x8000_0000 == 0 { 1.0 } else { -1.0 },
            steps: w[1] & 0x7FFF_FFFF,
        })
    }

    fn ensure_buffers(&mut self, n: usize) -> Result<(), String> {
        if self.bufs.as_ref().is_some_and(|b| b.cap >= n) {
            return Ok(());
        }
        let cap = n.next_power_of_two().min(MAX_DISPATCH_WALKS).max(n);
        let d = &self.device;
        let sz = |per: usize| (cap * per) as u64;
        if sz(std::mem::size_of::<GpuWalk>()) > self.max_buffer {
            return Err(format!("{cap} walks exceed the device buffer limit"));
        }
        let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let ubuf = buf(
            "params",
            std::mem::size_of::<Params>() as u64,
            U::UNIFORM | U::COPY_DST,
        );
        let walks = buf("walks", sz(std::mem::size_of::<GpuWalk>()), U::STORAGE);
        let counter = buf("running", 4, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let counter_rb = buf("running readback", 4, U::MAP_READ | U::COPY_DST);
        let input = buf("launches", sz(20), U::STORAGE | U::COPY_DST);
        let output = buf("outcomes", sz(8), U::STORAGE | U::COPY_SRC);
        let output_rb = buf("outcomes readback", sz(8), U::MAP_READ | U::COPY_DST);
        let entries = [
            (0, &ubuf),
            (1, &self.boxes),
            (2, &self.idx),
            (3, &walks),
            (4, &counter),
            (5, &input),
            (6, &output),
        ]
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding,
            resource: b.as_entire_binding(),
        });
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("walk"),
            layout: &self.layout,
            entries: &entries,
        });
        self.bufs = Some(Buffers {
            cap,
            ubuf,
            _walks: walks,
            counter,
            counter_rb,
            input,
            output,
            output_rb,
            bind,
        });
        Ok(())
    }

    fn run_chunk(&mut self, launches: &[Launch]) -> Result<Vec<Outcome>, String> {
        let n = launches.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        self.ensure_buffers(n)?;
        let shift = self.shift;
        let input: Vec<[u32; 5]> = launches
            .par_iter()
            .map(|l| {
                let p = rel(l.start, shift);
                #[allow(clippy::cast_possible_truncation)]
                [
                    p[0].to_bits(),
                    p[1].to_bits(),
                    p[2].to_bits(),
                    l.key as u32,
                    (l.key >> 32) as u32,
                ]
            })
            .collect();
        let bufs = self.bufs.as_ref().expect("allocated above");
        self.queue
            .write_buffer(&bufs.input, 0, bytemuck::cast_slice(&input));
        let mut params = self.params;
        #[allow(clippy::cast_possible_truncation)]
        {
            params.ctl[0] = n as u32;
        }
        let d = &self.device;
        #[allow(clippy::cast_possible_truncation)]
        let groups = (n as u32).div_ceil(WORKGROUP);
        let spd = self.params.ctl[1];
        // Every round advances each running walk by `spd` steps; after
        // max_steps / spd + 1 rounds every walk has finished or been aborted.
        let max_rounds = self.params.ctl[2] / spd + 2;
        let mut left = u32::MAX;
        for round in 0..max_rounds {
            params.ctl[3] = u32::from(round == 0);
            self.queue
                .write_buffer(&bufs.ubuf, 0, bytemuck::bytes_of(&params));
            self.queue
                .write_buffer(&bufs.counter, 0, &0u32.to_ne_bytes());
            let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bufs.bind, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            enc.copy_buffer_to_buffer(&bufs.counter, 0, &bufs.counter_rb, 0, 4);
            self.queue.submit([enc.finish()]);
            left = read::<u32>(d, &bufs.counter_rb, 4)?[0];
            if left == 0 {
                break;
            }
        }
        if left != 0 {
            return Err(format!("{left} walks still running after the step budget"));
        }
        let bytes = (n * 8) as u64;
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&bufs.output, 0, &bufs.output_rb, 0, bytes);
        self.queue.submit([enc.finish()]);
        let out: Vec<[u32; 2]> = read(d, &bufs.output_rb, bytes)?;
        let bad = std::sync::atomic::AtomicBool::new(false);
        let mut res = Vec::with_capacity(n);
        out.par_iter()
            .map(|w| {
                self.decode(*w).unwrap_or_else(|_| {
                    bad.store(true, std::sync::atomic::Ordering::Relaxed);
                    Outcome {
                        sink: None,
                        sign: 1.0,
                        steps: 0,
                    }
                })
            })
            .collect_into_vec(&mut res);
        if bad.into_inner() {
            let first = out.iter().find_map(|w| self.decode(*w).err());
            return Err(first.unwrap_or_else(|| "invalid walk status".into()));
        }
        Ok(res)
    }
}

/// Maps the first `bytes` of `buf` and copies them out.
fn read<T: Pod>(device: &wgpu::Device, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<T>, String> {
    let slice = buf.slice(..bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let out = bytemuck::cast_slice(&slice.get_mapped_range().map_err(|e| e.to_string())?).to_vec();
    buf.unmap();
    Ok(out)
}

/// Buffers reused across calls.
#[derive(Debug)]
struct Buffers {
    cap: usize,
    ubuf: wgpu::Buffer,
    _walks: wgpu::Buffer,
    counter: wgpu::Buffer,
    counter_rb: wgpu::Buffer,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    output_rb: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl WalkEngine for GpuEngine {
    fn run(&mut self, _scene: &Scene<'_>, launches: &[Launch]) -> Result<Vec<Outcome>, String> {
        let per = MAX_DISPATCH_WALKS
            .min((self.max_buffer / std::mem::size_of::<GpuWalk>() as u64) as usize)
            .max(1);
        let mut out = Vec::with_capacity(launches.len());
        for chunk in launches.chunks(per) {
            out.extend(self.run_chunk(chunk)?);
        }
        Ok(out)
    }

    fn preferred_walks(&self) -> usize {
        self.opts.walks_per_call
    }
}
