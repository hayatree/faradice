//! wgpu resources of the 3D view.

use iced::wgpu;
use iced::wgpu::util::DeviceExt;
use iced::{Rectangle, Size};

use super::SceneData;

/// One box instance (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Instance {
    pub(crate) min: [f32; 3],
    pub(crate) net: u32,
    pub(crate) max: [f32; 3],
    /// Bit 0: solve target.
    pub(crate) flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) light_dir: [f32; 4],
    pub(crate) selection: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

/// The 36 vertices of the unit cube, with face normals.
fn cube() -> Vec<Vertex> {
    // (normal, four corners counter-clockwise seen from outside)
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [1., 0., 0.],
            [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
        ),
        (
            [-1., 0., 0.],
            [[0., 1., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 1.]],
        ),
        (
            [0., 1., 0.],
            [[1., 1., 0.], [0., 1., 0.], [0., 1., 1.], [1., 1., 1.]],
        ),
        (
            [0., -1., 0.],
            [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
        ),
        (
            [0., 0., 1.],
            [[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
        ),
        (
            [0., 0., -1.],
            [[0., 1., 0.], [1., 1., 0.], [1., 0., 0.], [0., 0., 0.]],
        ),
    ];
    let mut v = Vec::with_capacity(36);
    for (n, c) in faces {
        for i in [0, 1, 2, 0, 2, 3] {
            v.push(Vertex {
                pos: c[i],
                normal: n,
            });
        }
    }
    v
}

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(crate) struct Pipeline {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    instances: Option<wgpu::Buffer>,
    count: u32,
    generation: u64,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth: Option<(Size<u32>, wgpu::TextureView)>,
}

impl Pipeline {
    fn create(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("faradice.boxes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("boxes.wgsl").into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("faradice.boxes.uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("faradice.boxes.layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("faradice.boxes.bind_group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("faradice.boxes.pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let instance_attrs =
            wgpu::vertex_attr_array![2 => Float32x3, 3 => Uint32, 4 => Float32x3, 5 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("faradice.boxes.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("faradice.boxes.cube"),
            contents: bytemuck::cast_slice(&cube()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            pipeline,
            vertices,
            instances: None,
            count: 0,
            generation: 0,
            uniforms,
            bind_group,
            depth: None,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_size: Size<u32>,
        data: Option<&SceneData>,
        uniforms: &Uniforms,
    ) {
        match data {
            Some(d) if d.generation != self.generation => {
                self.instances = (!d.instances.is_empty()).then(|| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("faradice.boxes.instances"),
                        contents: bytemuck::cast_slice(&d.instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    })
                });
                self.count = u32::try_from(d.instances.len()).unwrap_or(u32::MAX);
                self.generation = d.generation;
            }
            Some(_) => {}
            None => {
                self.instances = None;
                self.count = 0;
                self.generation = 0;
            }
        }
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
        let size = Size::new(target_size.width.max(1), target_size.height.max(1));
        if self.depth.as_ref().is_none_or(|(s, _)| *s != size) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("faradice.boxes.depth"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((
                size,
                tex.create_view(&wgpu::TextureViewDescriptor::default()),
            ));
        }
    }

    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip: Rectangle<u32>,
    ) {
        let (Some(instances), Some((_, depth))) = (&self.instances, &self.depth) else {
            return;
        };
        if clip.width == 0 || clip.height == 0 {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("faradice.boxes.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        #[allow(clippy::cast_precision_loss)]
        pass.set_viewport(
            clip.x as f32,
            clip.y as f32,
            clip.width as f32,
            clip.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(clip.x, clip.y, clip.width, clip.height);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.draw(0..36, 0..self.count);
    }
}

impl iced::widget::shader::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        Self::create(device, format)
    }
}
