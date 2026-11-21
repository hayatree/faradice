//! Convergence plot: relative error of the total capacitance versus walks,
//! log-log, with the 1/√N guide and the accuracy target (design D1 §4.4).

use iced::widget::canvas::{self, Frame, Path, Stroke, Text};
use iced::{Color, Point, Rectangle, Renderer, Size, Theme, mouse};

/// Data of one plot.
#[derive(Debug, Clone, Default)]
pub(crate) struct Convergence {
    /// `(walks, relative error at confidence σ)` samples, increasing walks.
    pub(crate) points: Vec<(u64, f64)>,
    /// Target relative tolerance (horizontal line), if known.
    pub(crate) target: Option<f64>,
}

const PAD: f32 = 36.0;

impl<M> canvas::Program<M> for Convergence {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut f = Frame::new(renderer, bounds.size());
        let pal = theme.extended_palette();
        let fg = pal.background.base.text;
        let faint = Color { a: 0.35, ..fg };
        let pts: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter(|(n, e)| *n > 0 && e.is_finite() && *e > 0.0)
            .map(|&(n, e)| ((n as f64).log10(), e.log10()))
            .collect();
        let label = |f: &mut Frame, s: String, p: Point| {
            f.fill_text(Text {
                content: s,
                position: p,
                color: faint,
                size: 11.0.into(),
                ..Text::default()
            });
        };
        if pts.len() < 2 {
            label(
                &mut f,
                "convergence plot: waiting for data".into(),
                Point::new(PAD, PAD),
            );
            return vec![f.into_geometry()];
        }
        let tgt = self.target.filter(|t| *t > 0.0).map(f64::log10);
        let (mut x0, mut x1) = (pts[0].0, pts[pts.len() - 1].0);
        let mut y0 = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let mut y1 = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        if let Some(t) = tgt {
            y0 = y0.min(t);
            y1 = y1.max(t);
        }
        x0 = x0.floor();
        x1 = x1.ceil().max(x0 + 1.0);
        y0 = y0.floor();
        y1 = y1.ceil().max(y0 + 1.0);
        let Size { width, height } = bounds.size();
        let (w, h) = ((width - 2.0 * PAD).max(1.0), (height - 2.0 * PAD).max(1.0));
        #[allow(clippy::cast_possible_truncation)]
        let map = |x: f64, y: f64| {
            Point::new(
                PAD + ((x - x0) / (x1 - x0)) as f32 * w,
                PAD + (1.0 - ((y - y0) / (y1 - y0)) as f32) * h,
            )
        };
        // Axes with decade ticks.
        f.stroke(
            &Path::rectangle(Point::new(PAD, PAD), Size::new(w, h)),
            Stroke::default().with_color(faint).with_width(1.0),
        );
        #[allow(clippy::cast_possible_truncation)]
        for d in (x0 as i32)..=(x1 as i32) {
            let p = map(f64::from(d), y0);
            label(&mut f, format!("1e{d}"), Point::new(p.x - 10.0, p.y + 6.0));
        }
        #[allow(clippy::cast_possible_truncation)]
        for d in (y0 as i32)..=(y1 as i32) {
            let p = map(x0, f64::from(d));
            label(
                &mut f,
                format!("{}%", 100.0 * 10f64.powi(d)),
                Point::new(2.0, p.y - 6.0),
            );
        }
        label(
            &mut f,
            "walks".into(),
            Point::new(PAD + w - 30.0, PAD + h + 18.0),
        );
        // Accuracy target.
        if let Some(t) = tgt {
            f.stroke(
                &Path::line(map(x0, t), map(x1, t)),
                Stroke::default()
                    .with_color(pal.success.base.color)
                    .with_width(1.0),
            );
        }
        // 1/√N guide through the last point.
        let (lx, ly) = pts[pts.len() - 1];
        let guide = |x: f64| ly - 0.5 * (x - lx);
        f.stroke(
            &Path::line(map(x0, guide(x0)), map(x1, guide(x1))),
            Stroke::default().with_color(faint).with_width(1.0),
        );
        // The series.
        let path = Path::new(|b| {
            b.move_to(map(pts[0].0, pts[0].1));
            for &(x, y) in &pts[1..] {
                b.line_to(map(x, y));
            }
        });
        f.stroke(
            &path,
            Stroke::default()
                .with_color(pal.primary.base.color)
                .with_width(2.0),
        );
        vec![f.into_geometry()]
    }
}
