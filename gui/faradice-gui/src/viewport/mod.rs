//! 3D view of the solved geometry, drawn with wgpu inside the iced frame
//! through the `shader` widget (design D1 §2.4).
//!
//! Box instances are uploaded once per scene (keyed by a generation number);
//! camera and selection changes only update a small uniform buffer, so
//! orbiting a 10⁶-box layout does not copy geometry (gate G3).

mod pipeline;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use faradice_geom::Geometry;
use glam::{Mat4, Vec3};
use iced::widget::shader::{self, Action, Viewport};
use iced::{Event, Point, Rectangle, mouse, wgpu};

pub(crate) use pipeline::Instance;

/// Geometry uploaded to the GPU.
#[derive(Debug)]
pub(crate) struct SceneData {
    generation: u64,
    instances: Vec<Instance>,
    /// Bounding sphere (scene coordinates are centred, µm).
    radius: f32,
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

/// What the view shows: shared geometry and the current selection.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scene {
    data: Option<Arc<SceneData>>,
    /// Net id (index into the geometry's nets) to highlight.
    pub(crate) selected: Option<u32>,
    /// Dim nets that are not solve targets.
    pub(crate) dim_non_targets: bool,
    /// Extra yaw (radians) added to the camera, for the frame benchmark.
    pub(crate) spin: f32,
}

impl Scene {
    /// Builds instances from `geom`; `is_target(net index)` marks targets.
    pub(crate) fn new(geom: &Geometry, is_target: impl Fn(usize) -> bool) -> Self {
        let b = geom.bounds();
        let (lo, hi) = (b.min(), b.max());
        let c = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        #[allow(clippy::cast_possible_truncation)]
        let f = |v: f64| v as f32;
        let instances = geom
            .boxes()
            .iter()
            .enumerate()
            .map(|(i, bx)| {
                let (a, z) = (bx.min(), bx.max());
                #[allow(clippy::cast_possible_truncation)]
                let net = geom.box_net(faradice_core::BoxId::new(i as u32)).index();
                #[allow(clippy::cast_possible_truncation)]
                Instance {
                    min: [f(a[0] - c[0]), f(a[1] - c[1]), f(a[2] - c[2])],
                    net: net as u32,
                    max: [f(z[0] - c[0]), f(z[1] - c[1]), f(z[2] - c[2])],
                    flags: u32::from(is_target(net)),
                }
            })
            .collect();
        let radius = 0.5
            * f(
                ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2))
                    .sqrt(),
            );
        Self {
            data: Some(Arc::new(SceneData {
                generation: GENERATION.fetch_add(1, Ordering::Relaxed),
                instances,
                radius: radius.max(1e-3),
            })),
            selected: None,
            dim_non_targets: true,
            spin: 0.0,
        }
    }

    /// Number of boxes.
    pub(crate) fn len(&self) -> usize {
        self.data.as_ref().map_or(0, |d| d.instances.len())
    }
}

/// Orbit camera, kept in the widget state (no app messages while dragging).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Camera {
    yaw: f32,
    pitch: f32,
    /// Distance in units of the scene radius.
    zoom: f32,
    pan: Vec3,
    drag: Option<(Point, mouse::Button)>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: -0.9,
            pitch: 0.55,
            zoom: 2.6,
            pan: Vec3::ZERO,
            drag: None,
        }
    }
}

impl Camera {
    fn eye_dir(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
    }

    fn view_proj(&self, radius: f32, aspect: f32) -> Mat4 {
        let dist = self.zoom * radius;
        let eye = self.pan + self.eye_dir() * dist;
        let view = Mat4::look_at_rh(eye, self.pan, Vec3::Z);
        let near = (dist - 1.5 * radius).max(dist * 1e-3);
        let far = dist + 1.5 * radius;
        Mat4::perspective_rh(0.7, aspect.max(1e-3), near, far) * view
    }
}

impl<M> shader::Program<M> for Scene {
    type State = Camera;
    type Primitive = Primitive;

    fn update(
        &self,
        cam: &mut Camera,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<M>> {
        let Event::Mouse(e) = event else { return None };
        match e {
            mouse::Event::ButtonPressed(
                b @ (mouse::Button::Left | mouse::Button::Right | mouse::Button::Middle),
            ) => {
                let p = cursor.position_in(bounds)?;
                cam.drag = Some((p, *b));
                Some(Action::capture())
            }
            mouse::Event::ButtonReleased(_) if cam.drag.is_some() => {
                cam.drag = None;
                Some(Action::capture())
            }
            mouse::Event::CursorMoved { position } => {
                let (last, button) = cam.drag?;
                let p = Point::new(position.x - bounds.x, position.y - bounds.y);
                let (dx, dy) = (p.x - last.x, p.y - last.y);
                cam.drag = Some((p, button));
                if button == mouse::Button::Left {
                    cam.yaw -= dx * 0.008;
                    cam.pitch = (cam.pitch + dy * 0.008).clamp(-1.55, 1.55);
                } else {
                    let r = self.data.as_ref().map_or(1.0, |d| d.radius);
                    let s = cam.zoom * r / bounds.height.max(1.0);
                    let fwd = -cam.eye_dir();
                    let right = fwd.cross(Vec3::Z).normalize_or_zero();
                    let up = right.cross(fwd);
                    cam.pan += (-right * dx + up * dy) * s;
                }
                Some(Action::request_redraw().and_capture())
            }
            mouse::Event::WheelScrolled { delta } => {
                cursor.position_in(bounds)?;
                let y = match delta {
                    mouse::ScrollDelta::Lines { y, .. } => *y,
                    mouse::ScrollDelta::Pixels { y, .. } => *y / 40.0,
                };
                cam.zoom = (cam.zoom * (1.0 - 0.12 * y)).clamp(0.05, 50.0);
                Some(Action::request_redraw().and_capture())
            }
            _ => None,
        }
    }

    fn draw(&self, cam: &Camera, _cursor: mouse::Cursor, bounds: Rectangle) -> Primitive {
        let mut cam = *cam;
        cam.yaw += self.spin;
        let radius = self.data.as_ref().map_or(1.0, |d| d.radius);
        let view_proj = cam.view_proj(radius, bounds.width / bounds.height.max(1.0));
        let light = (cam.eye_dir() + Vec3::new(0.3, -0.2, 0.8)).normalize();
        Primitive {
            data: self.data.clone(),
            uniforms: pipeline::Uniforms {
                view_proj: view_proj.to_cols_array_2d(),
                light_dir: [light.x, light.y, light.z, 0.0],
                selection: [
                    self.selected.unwrap_or(0),
                    u32::from(self.selected.is_some()),
                    u32::from(self.dim_non_targets),
                    0,
                ],
            },
        }
    }

    fn mouse_interaction(
        &self,
        cam: &Camera,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if cam.drag.is_some() {
            mouse::Interaction::Grabbing
        } else if cursor.is_over(bounds) {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::default()
        }
    }
}

/// One frame of the view.
#[derive(Debug)]
pub(crate) struct Primitive {
    data: Option<Arc<SceneData>>,
    uniforms: pipeline::Uniforms,
}

impl shader::Primitive for Primitive {
    type Pipeline = pipeline::Pipeline;

    fn prepare(
        &self,
        pipeline: &mut pipeline::Pipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        pipeline.prepare(
            device,
            queue,
            viewport.physical_size(),
            self.data.as_deref(),
            &self.uniforms,
        );
    }

    fn render(
        &self,
        pipeline: &pipeline::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        pipeline.render(encoder, target, *clip_bounds);
    }
}
