#!/bin/sh
# Prepares the patched crates used by the GUI workspace (`[patch.crates-io]`
# in gui/Cargo.toml). Each crate is downloaded from crates.io with a pinned
# checksum, then patched.
#
# Why: faradice uses no GPL-family code (docs/design/D1_gui.md §2.1). Two
# crates in the iced Wayland graph (winit: Apache-2.0; wayland-protocols-misc:
# MIT) depend on or bundle KDE protocol files under LGPL-2.1-or-later and
# generate Rust bindings from them. Licenses of the patches: vendor/NOTICE.md.
#
# - winit 0.30.13 depends on wayland-protocols-plasma for the optional KDE
#   window blur effect (blur.xml). The patch removes the effect and that
#   dependency.
# - wayland-protocols-misc 0.3.12 always compiles the KDE server-decoration
#   protocol (server-decoration.xml), which nothing in the graph uses. The
#   file is deleted and the patch removes its module.
#
# Remove an entry once a released upstream version makes these optional.
set -eu
here=$(cd "$(dirname "$0")" && pwd)

# fetch NAME VERSION SHA256 PATCH [FILES TO DELETE...]
fetch() {
    name=$1 version=$2 sha=$3 patchfile=$4
    shift 4
    dest="$here/$name"
    stamp="$dest/.faradice-patched"
    want="$version $sha $(sha256sum "$here/$patchfile" | cut -d' ' -f1)"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
        return 0
    fi
    tmp=$(mktemp -d)
    curl --fail --silent --show-error --location \
        "https://static.crates.io/crates/$name/$name-$version.crate" -o "$tmp/c.crate"
    echo "$sha  $tmp/c.crate" | sha256sum --check --quiet
    tar -xzf "$tmp/c.crate" -C "$tmp"
    for f in "$@"; do
        rm "$tmp/$name-$version/$f"
    done
    (cd "$tmp/$name-$version" && patch --quiet -p1 < "$here/$patchfile")
    rm -rf "$dest"
    mv "$tmp/$name-$version" "$dest"
    rm -rf "$tmp"
    echo "$want" > "$stamp"
    echo "prepared vendor/$name $version"
}

fetch winit 0.30.13 \
    a6755fa58a9f8350bd1e472d4c3fcc25f824ec358933bba33306d0b63df5978d \
    winit-0.30.13-no-kde-blur.patch
fetch wayland-protocols-misc 0.3.12 \
    6e9567599ef23e09b8dad6e429e5738d4509dfc46b3b21f32841a304d16b29c8 \
    wayland-protocols-misc-0.3.12-no-kde-decoration.patch \
    protocols/server-decoration.xml
