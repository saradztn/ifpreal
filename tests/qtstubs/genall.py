"""Generate stubs for every Qt binary, plugins included.

Scanning only the Qt libraries missed symbols the platform plugins import on
their own -- libqoffscreen.so needs glXMakeCurrent, which libQt6Gui never
mentions.  The linker resolves at load time, so the scan has to cover every
object Qt will dlopen, not just the ones imported by Python.
"""
import os, re, subprocess
from collections import defaultdict
QTDIR="/home/user/ifpreal/.venv/lib/python3.11/site-packages/PySide6/Qt"
WANT={"libGL.so.1":    re.compile(r"\bgl[A-Z]\w+"),
      "libEGL.so.1":   re.compile(r"\begl[A-Z]\w+"),
      "libxkbcommon.so.0": re.compile(r"\bxkb_[A-Za-z0-9_]+"),
      "libdbus-1.so.3":    re.compile(r"\bdbus_[A-Za-z0-9_]+")}
found=defaultdict(set)
for root,_,files in os.walk(QTDIR):
    for name in files:
        if not name.endswith(".so") and ".so." not in name: continue
        path=os.path.join(root,name)
        try:
            out=subprocess.run(["nm","-D","--undefined-only",path],
                               capture_output=True,text=True,timeout=30).stdout
        except Exception: continue
        for line in out.splitlines():
            parts=line.split()
            if not parts: continue
            sym=parts[-1].split("@")[0]
            for lib,pat in WANT.items():
                if pat.fullmatch(sym): found[lib].add(sym)
for lib,syms in found.items():
    body=["/* auto-generated stub for headless Qt testing */","#include <stddef.h>"]
    body+= [f"long {s}(void){{return 0;}}" for s in sorted(syms)]
    src=f"/tmp/glstubs/{lib}.c"; open(src,"w").write("\n".join(body)+"\n")
    print(f"{lib}: {len(syms)} symbols")
