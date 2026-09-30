# GTA FBX → IFP Converter

A real FBX → GTA San Andreas / MTA:SA IFP retarget engine. This repository is
being built in the order the format demands — real data first, code second —
and every claim below is backed by a measurement against a real file.

## Verified so far

| Area | Status | Evidence |
| --- | --- | --- |
| Binary FBX reader (FBX 7.4) | working | `testdata/samba_dancing.fbx`: 5 371 nodes, 0 resyncs, 0.18 s |
| FBX source rig + clips | working | 1 armature (67 bones), 1 clip (18.2 s, 1 149 keys, 52 animated bones) |
| FBX transforms | working | agrees with three.js `FBXLoader` to 4.5e-6° over 2 869 rotations |
| Coordinate conversion | working | derived from declared bases, maps FBX up(+Y)→GTA up(+Z) |
| Target DFF → HAnim skeleton | working | `testdata/male01.dff`, `claude.dff`, `army.dff` |
| Windows CLI (`setup.py`, `cli.py`) | working | refuses `convert` rather than faking it |
| Bone mapping, rest-pose solver, retarget, bake | not started | — |
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

**FBX Euler angles are degrees, and the default order is `ZYX`.** All three of
these are only checkable against an independent implementation, because each
one produces output that looks entirely plausible:

* The transform lives in the node's `Properties70` block, not its property
  list. Reading the legacy names positionally returns zeros and collapses the
  whole rig onto the origin.
* Every Euler angle is in degrees, in both `Properties70` and the animation
  curves. This file stores rotations up to 765°; read as radians, 765° becomes
  13.4 rad and the animation folds the skeleton through itself.
* The FBX SDK's default rotation order is `ZYX`, not `XYZ`, and the order
  letters index the array by *axis name*, not by position. This file declares
  no `RotationOrder` at all, so the fallback is what every joint uses.

Together they put the pelvis at a ~90° average tilt and the head below the
hips for a third of the clip. Fixed, the pelvis tilt is 10.5° and the head
stays above the hips for 100% of frames — matching three.js exactly.

**A time a few microseconds below a key interpolated the wrong key.** FBX
stores key times in seconds, so a time read back from a float32 field lands
slightly off. `(t - t_lo) / span` then loses most of its significant digits to
cancellation, the result rounds back to the previous key's value, and the pose
silently snaps to the wrong frame. Keys are now matched to within the
precision float32 seconds actually delivers.

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

## Windows install and CLI

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\gtafbx doctor
```

`setup.py` installs two equivalent console entry points, `gtafbx` and
`gta-fbx-ifp`, plus the GUI entry point once it exists. You can also run the
tool straight from a checkout without installing, with `py -3 cli.py <command>`.

```
gtafbx doctor                       environment + which pipeline stages exist
gtafbx inspect-fbx dance.fbx        source rig, clips, coordinate conversion
gtafbx inspect-dff male01.dff       target ped skeleton and HAnim resolution
gtafbx check dance.fbx male01.dff   can this pair be trusted?
gtafbx convert --fbx ... --dff ... --out dance.ifp
```

`--json` prints a machine-readable report on stdout and is accepted on either
side of the subcommand; `--quiet` suppresses the human report; `--no-color`
drops ANSI colour. Exit codes: `0` ok, `2` usage, `3` bad input file, `4`
stage not implemented, `5` failed validation.

`convert` currently **refuses** and exits `4` without writing a file. Bone
mapping, the rest-pose solver, retargeting, the bake, the ANP3 writer and the
round-trip validator are not written yet, and the tool says so rather than
producing a file that looks like a conversion.

## Running the tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

The FBX layer is checked against a dump of three.js `FBXLoader`'s own
evaluation of the same file; see `testdata/reference/README.md`.
