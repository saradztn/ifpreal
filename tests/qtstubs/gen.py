"""Generate a stub .so for every system library Qt is missing.

Each exported function returns 0 (NULL). That is the "this platform is not
available" answer for the EGL, xkbcommon and D-Bus entry points, which is
exactly right for a headless test run: Qt falls back to its software paths
instead of trying to talk to a graphics device that is not there.
"""
import os, re, subprocess, sys
from collections import defaultdict

QTLIB = "/home/user/ifpreal/.venv/lib/python3.11/site-packages/PySide6/Qt/lib"
# Undefined symbols are grouped by which library actually provides them.
WANT = {
    "libGL.so.1":    re.compile(r"\bgl(?:X|Get|Set|Create|Delete|Gen|Bind|Enable|Disable|"
                               r"Clear|Color|Draw|Read|Flush|Finish|Pixel|Stencil|Depth|Blend|"
                               r"Logic|Point|Line|Polygon|Scissor|Viewport|Front|Tex|Matrix|Model|"
                               r"Proj|Attrib|Light|Material|Fog|Alpha|Accum|Stack|Map|Unmap|"
                               r"Render|Shader|Vertex|Buffer|Frame|Clip|List|Query|Get|Is|Sync|"
                               r"Hint|Error|String|Integerv|Floatv|Booleanv)\w*"),
    "libEGL.so.1":   re.compile(r"\begl[A-Z]\w+"),
    "libxkbcommon.so.0": re.compile(r"\bxkb_[a-z_]+"),
    "libdbus-1.so.3":    re.compile(r"\bdbus_[a-z_]+"),
}

found = defaultdict(set)
for name in sorted(os.listdir(QTLIB)):
    if not name.startswith("libQt6") or not name.endswith(".so.6"):
        continue
    path = os.path.join(QTLIB, name)
    try:
        out = subprocess.run(["nm", "-D", "--undefined-only", path],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        continue
    for line in out.splitlines():
        sym = line.split()[-1] if line.split() else ""
        for lib, pattern in WANT.items():
            if sym and pattern.fullmatch(sym):
                found[lib].add(sym)

for lib, syms in found.items():
    if not syms:
        continue
    body = ["/* auto-generated stub; see gen.py */", "#include <stddef.h>"]
    for s in sorted(syms):
        # void return and NULL are the honest answers for "not available";
        # the exact prototype does not matter because nothing is called.
        body.append(f"long {s}(void) {{ return 0; }}")
    src = f"/tmp/glstubs/{lib}.c"
    with open(src, "w") as fh:
        fh.write("\n".join(body) + "\n")
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-w", "-o", f"/tmp/glstubs/{lib}", src],
                       capture_output=True, text=True)
    print(f"{lib}: {len(syms)} symbols -> {'ok' if r.returncode == 0 else r.stderr[:200]}")
