"""Rebuild each stub with the version node names the Qt libs actually ask for.

Guessing these gets the ld.so "version not found" abort, so they are read out
of the ELF version-requirement tables instead of typed in.
"""
import os, re, subprocess
from collections import defaultdict
QTLIB="/home/user/ifpreal/.venv/lib/python3.11/site-packages/PySide6/Qt/lib"
STUBS={"libGL.so.1","libEGL.so.1","libxkbcommon.so.0","libdbus-1.so.3"}
needed=defaultdict(set)
for name in sorted(os.listdir(QTLIB)):
    if not (name.startswith("libQt6") and name.endswith(".so.6")): continue
    path=os.path.join(QTLIB,name)
    out=subprocess.run(["readelf","-V",path],capture_output=True,text=True).stdout
    current=None
    for line in out.splitlines():
        m=re.search(r"File:\s+(\S+)",line)
        if m: current=m.group(1)
        m=re.search(r"Name:\s+(\S+)\s+Flags",line)
        if m and current in STUBS: needed[current].add(m.group(1))
for so,versions in needed.items():
    c=f"/tmp/glstubs/{so}.c"
    if not os.path.exists(c): continue
    if not versions:
        r=subprocess.run(["gcc","-shared","-fPIC","-w","-o",so,c],capture_output=True,text=True)
        print(f"{so}: no version needed")
        continue
    body="\n".join(f"{v} {{ global: *; }};" for v in sorted(versions))
    open(f"/tmp/glstubs/{so}.map","w").write(body+"\n")
    r=subprocess.run(["gcc","-shared","-fPIC","-w","-o",so,c,
                      f"-Wl,--version-script=/tmp/glstubs/{so}.map"],
                     capture_output=True,text=True)
    print(f"{so}: versions {sorted(versions)} -> {'ok' if r.returncode==0 else r.stderr[:200]}")
