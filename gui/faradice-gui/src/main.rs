//! `faradice-gui`: desktop front end of faradice (design D1).
//!
//! The GUI is a thin client of `faradice-engine`: it loads a [`Project`],
//! builds a [`RunRequest`] from the user's choices, runs it as a background
//! [`Job`] and folds the job's events into its view state. It never calls
//! the solver directly, so CLI and GUI results are identical.
//!
//! [`Project`]: faradice_engine::Project
//! [`RunRequest`]: faradice_engine::RunRequest
//! [`Job`]: faradice_engine::Job

mod app;
mod cli;
mod matrix;
mod plot;
mod table;
mod viewport;

use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: faradice-gui [STRUCTURE.faradice.toml | LAYOUT.gds --tech TECH.toml] [-c CONFIG.toml]"
    );
    std::process::exit(2);
}

/// Command-line arguments (all optional; everything can be chosen in the GUI).
#[derive(Debug, Default, Clone)]
pub(crate) struct Args {
    pub(crate) input: Option<PathBuf>,
    pub(crate) tech: Option<PathBuf>,
    pub(crate) config: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args_os().skip(1);
    while let Some(arg) = it.next() {
        match arg.to_str() {
            Some("--tech") => a.tech = Some(it.next().unwrap_or_else(|| usage()).into()),
            Some("-c" | "--config") => {
                a.config = Some(it.next().unwrap_or_else(|| usage()).into());
            }
            Some("-h" | "--help") => usage(),
            Some(s) if s.starts_with('-') => usage(),
            _ if a.input.is_none() => a.input = Some(arg.into()),
            _ => usage(),
        }
    }
    a
}

fn main() -> iced::Result {
    let args = parse_args();
    iced::application(
        move || app::App::boot(args.clone()),
        app::App::update,
        app::App::view,
    )
    .title(app::App::title)
    .subscription(app::App::subscription)
    .theme(app::App::theme)
    .window_size((1280.0, 820.0))
    .antialiasing(true)
    .run()
}
