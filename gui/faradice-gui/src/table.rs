//! A virtualized table: only the rows inside the scroll viewport are built,
//! so 10⁵-row coupling tables stay cheap (design D1 gate G2).

use iced::widget::{Column, Row, button, column, container, scrollable, space, text};
use iced::{Element, Fill, Length, Theme};

/// Fixed row height in logical pixels.
pub(crate) const ROW_HEIGHT: f32 = 26.0;
/// Rows built above and below the visible range.
const OVERSCAN: usize = 8;

/// Scroll state of one table.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Scroll {
    /// Vertical offset (px).
    pub(crate) offset: f32,
    /// Viewport height (px); 0 until the first scroll event.
    pub(crate) height: f32,
}

impl Scroll {
    /// Updates from a scrollable viewport.
    pub(crate) fn set(&mut self, v: &scrollable::Viewport) {
        self.offset = v.absolute_offset().y;
        self.height = v.bounds().height;
    }

    /// The row range to build for `len` rows.
    pub(crate) fn visible(&self, len: usize) -> std::ops::Range<usize> {
        // Before the first event the height is unknown: assume a tall window.
        let height = if self.height > 0.0 {
            self.height
        } else {
            1600.0
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let first = (self.offset / ROW_HEIGHT).floor().max(0.0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let count = (height / ROW_HEIGHT).ceil() as usize + 1;
        let start = first.saturating_sub(OVERSCAN).min(len);
        let end = (first + count + OVERSCAN).min(len);
        start..end
    }
}

/// A column definition: header and relative width.
pub(crate) struct Col {
    pub(crate) title: &'static str,
    pub(crate) width: Length,
    pub(crate) right: bool,
}

pub(crate) const fn col(title: &'static str, portion: u16, right: bool) -> Col {
    Col {
        title,
        width: Length::FillPortion(portion),
        right,
    }
}

/// Row state of one table.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Rows {
    /// Number of rows.
    pub(crate) len: usize,
    /// Scroll state.
    pub(crate) scroll: Scroll,
    /// Highlighted row.
    pub(crate) selected: Option<usize>,
    /// Scrollable id (for scroll operations).
    pub(crate) id: Option<&'static str>,
}

fn cell<'a, M: 'a>(s: String, c: &Col) -> Element<'a, M> {
    let t = text(s)
        .size(13)
        .wrapping(iced::widget::text::Wrapping::None);
    let t = if c.right {
        t.align_x(iced::alignment::Horizontal::Right)
    } else {
        t
    };
    container(t.width(Fill))
        .width(c.width)
        .padding([0, 6])
        .into()
}

/// Builds the table. `cells(i)` returns the cell texts of row `i`;
/// `on_select(i)` makes rows clickable.
pub(crate) fn view<'a, M: Clone + 'a>(
    cols: &[Col],
    rows: Rows,
    cells: impl Fn(usize) -> Vec<String>,
    on_select: impl Fn(usize) -> M,
    on_scroll: impl Fn(scrollable::Viewport) -> M + 'a,
) -> Element<'a, M> {
    let Rows {
        len,
        scroll,
        selected,
        id,
    } = rows;
    let header =
        Row::with_children(cols.iter().map(|c| cell(c.title.to_owned(), c))).height(ROW_HEIGHT);
    let range = scroll.visible(len);
    #[allow(clippy::cast_precision_loss)]
    let top = range.start as f32 * ROW_HEIGHT;
    #[allow(clippy::cast_precision_loss)]
    let bottom = (len - range.end) as f32 * ROW_HEIGHT;
    let mut body = Column::new().push(space().height(top));
    for i in range {
        let r = Row::with_children(cells(i).into_iter().zip(cols).map(|(s, c)| cell(s, c)))
            .height(ROW_HEIGHT)
            .align_y(iced::Center);
        let is_sel = selected == Some(i);
        body = body.push(
            button(r)
                .padding(0)
                .width(Fill)
                .on_press(on_select(i))
                .style(move |theme: &Theme, status| {
                    let base = if is_sel {
                        button::primary(theme, status)
                    } else {
                        button::text(theme, status)
                    };
                    button::Style {
                        border: iced::Border::default(),
                        ..base
                    }
                }),
        );
    }
    body = body.push(space().height(bottom));
    let s = scrollable(body)
        .on_scroll(on_scroll)
        .height(Fill)
        .width(Fill);
    let s = match id {
        Some(id) => s.id(id),
        None => s,
    };
    column![container(header).style(container::rounded_box), s].into()
}

/// Two-line hint shown in place of an empty table.
pub(crate) fn empty<'a, M: 'a>(title: &'a str, hint: &'a str) -> Element<'a, M> {
    container(column![text(title).size(18), text(hint).size(13)].spacing(6))
        .center(Fill)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_range_is_bounded_and_covers_the_viewport() {
        let s = Scroll {
            offset: 26_000.0,
            height: 520.0,
        };
        let r = s.visible(100_000);
        assert!(r.start <= 1000 && r.end >= 1020, "{r:?}");
        assert!(r.len() < 60);
        assert_eq!(s.visible(10), 10..10);
        assert_eq!(Scroll::default().visible(5), 0..5);
    }
}
