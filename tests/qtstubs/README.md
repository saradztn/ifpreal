# Headless Qt stub libraries

PySide6's `libQt6Gui` links against `libGL`, `libEGL`, `libxkbcommon` and
`libdbus-1` even when it will never call them. On a bare container none of
those are installed, so importing PySide6 fails with `ImportError: libGL.so.1`
and the window's tests cannot run at all.

The `.so` files here satisfy the dynamic linker. Every entry point returns
0/NULL, which is the correct "this platform is not available" answer: Qt then
takes its software rendering path, which is what the offscreen platform
plugin wants anyway.

They are only on `LD_LIBRARY_PATH` when the real libraries are missing, so on
a normal desktop nothing here is ever loaded. Regenerate with:

    python genall.py    # collect undefined symbols from every Qt binary
    python fixver.py    # add the version nodes the ELF requires
