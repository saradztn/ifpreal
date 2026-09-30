# GTA FBX → IFP Converter

A real FBX → GTA San Andreas / MTA:SA IFP retarget engine. This repository is
being built in the order the format demands — real data first, code second —
and every claim below is backed by a measurement against a real file.

## Verified so far

| Area | Status | Evidence |
| --- | --- | --- |
| Binary FBX reader (FBX 7.4) | working | `testdata/samba_dancing.fbx`: 5 371 nodes, 0 resyncs, 0.18 s |
| FBX source rig + clips | working | 1 armature (67 bones), 1 clip (18.2 s, 1 149 keys, 52 animated bones) |
| Coordinate conversion | working | derived from declared bases, maps FBX up(+Y)→GTA up(+Z) |
| Target DFF → HAnim skeleton | working | `testdata/male01.dff`, `claude.dff`, `army.dff` |
| Bone mapping, rest-pose solver, retarget, bake | in progress | — |
| ANP3 writer, validator, GUI, MTA resource | not started | — |

## Test data

* `testdata/samba_dancing.fbx` — binary FBX 7400, a real Mixamo rig with 18.2 s
  of animation.
* `testdata/male01.dff`, `claude.dff`, `army.dff`, `ballas1.dff`, `copgrl3.dff` —
  real GTA San Andreas ped DFFs (backups of the shipped skins), each with a
  33-frame hierarchy, a 32-bone HAnim plugin and a 28-bone skin.

## Two findings that shaped the design

**The FBX binary record header.** `PropertyListLen` has to be read from the
record header itself. Recovering it by a back-reference into the name bytes
silently corrupts every record after the first: the exact-fit check that
chooses the array-header width never passes, and the reader rewinds to the
start of the property list. On the Samba fixture that turned a 0.18 s clean
parse into a 3.0 s parse with a resync and curves with no key times.

**The DFF's HAnim `node_id` array is not trustworthy on its own.** The shipped
SA ped DFFs store their frames in an authoring order whose parent tree is
*isomorphic* to the canonical ped hierarchy, while the HAnim `node_id` array is
the canonical bone list left over in canonical order. 0 of 32 frames agree:

```
frame  3  'R Thigh'    node_id  3  -> canonical "Spine1"
frame 17  'L UpperArm' node_id 23  -> canonical "R Forearm"
frame 24  'R Hand'     node_id 41  -> canonical "L Thigh"
```

Trusting `node_id` would write IFP tracks that move the thigh when the head
track fires. `gta/tag_resolve.py` keeps the DFF in charge of the tag *set*,
the hierarchy and the bind transforms, resolves each frame's identity from
name + structure + side, and reports every disagreement instead of hiding it.

## Layout

```
gta_fbx_ifp_converter/
  core/     mathx.py (quaternions, XYZW), space.py (basis detection/conversion)
  fbx/      binary_reader.py, document.py, source_rig.py
  gta/      dff_reader.py, tag_resolve.py, bones.py
tests/      unit tests over the real files above
```

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```
