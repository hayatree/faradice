# Third-party patches

The two `.patch` files in this directory modify third-party crates. `fetch.sh` downloads
each crate from crates.io at a pinned version and checksum, applies its patch, and writes
the result to an ignored directory (`vendor/<crate>/`) that `gui/Cargo.toml` uses through
`[patch.crates-io]`. The patched crates are not redistributed in this repository. The
patches themselves contain short excerpts of the upstream sources (diff context) and
remain under the upstream licenses below, not under faradice's MIT OR Apache-2.0.

| Patch | Upstream | Version | License | Copyright | Change |
|---|---|---|---|---|---|
| `winit-0.30.13-no-kde-blur.patch` | [winit](https://github.com/rust-windowing/winit) | 0.30.13 | Apache-2.0 | The winit contributors | Removes the optional `wayland-protocols-plasma` dependency, so `Window::set_blur` is a no-op on Wayland |
| `wayland-protocols-misc-0.3.12-no-kde-decoration.patch` | [wayland-rs](https://github.com/smithay/wayland-rs) (`wayland-protocols-misc`) | 0.3.12 | MIT | Copyright (c) 2015 Elinor Berger | Removes the module generated from the KDE `server-decoration` protocol (its XML file is deleted by `fetch.sh`) |

Why: faradice uses no GPL-family code, including files bundled in dependencies. Both
crates bundle or depend on KDE protocol files licensed LGPL-2.1-or-later. See
[`../README.md`](../README.md#license-policy-and-the-patched-crates).
