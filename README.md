# GTA FBX → IFP Converter

Converts an animated FBX rig into a GTA: San Andreas / MTA:SA `.ifp`
animation file for a chosen ped skeleton, and tells you how well it did.

Both a window and a command line, over the same pipeline.

```
python cli.py convert --fbx dance.fbx --dff male01.dff --out dance.ifp
```

On the bundled Samba clip that is about five seconds and prints:

```
PASSED VALIDATION
  25 bones, 13925 keys; angle max 0.0265 deg, mean 0.0129, rms 0.0136
    [ 32] ' L UpperArm'          max 0.0265 deg at key 242 (mean 0.0132, rms 0.0139)
    [ 43] ' L Foot'              max 0.0257 deg at key 420 (mean 0.0134, rms 0.0140)
```

## What the numbers mean, and what they do not

Read this before believing any figure, including the ones above.

Those angles are the error introduced by **the IFP format itself** —
quantising a quaternion into `round(component * 4096)` and back. They are
measured by writing the file, reading it off disk, and comparing the decoded
frames to the data handed to the writer. They are not a claim about how close
the *source* motion is to the output.

Source-to-output fidelity is reported separately, on the same page:

- `unrepresentable` — source bones with no IFP equivalent. On the Samba rig
  that is 42 of 69 bones: Mixamo fingers have nothing to map to, and the
  mapper says so by name rather than dropping them silently.
- `time` — IFP stores time in 1/50 s steps. A 60 fps source cannot keep every
  key whatever this tool does. The motion is **resampled** onto those slots
  rather than decimated: every stored key is the source animation evaluated at
  that time, slerped between its own keys, so the keys lie on the path the
  motion took. The Samba clip's 1149 keys per track become 557, and the
  export says so.

  The difference is not small. On a limb rotating at 720°/s, keeping whichever
  source key claimed each slot is up to **14.4°** off the true motion, because
  the motion between that key and the next one is thrown away and drawn as a
  straight line. Resampling is exact on the same fixture. The two approaches
  produce the same number of keys and the same file size; only one of them is
  the animation.

**Reducing keys is a separate, opt-in, lossy thing.** It is off by default and
only runs with an explicit threshold, because it is the one step in the
pipeline that discards information:

    gtafbx convert --fbx dance.fbx --dff male01.dff --out dance.ifp \
        --reduce-keys 1.5

It reports what it cost in the terms a viewer would notice — how far the pose
at the worst instant now sits from what it was:

```
warn key reduction was asked for and applied: 5510 of 13925 keys moved the
pose by less than 1.50 degrees and were dropped; the pose is now up to
16.333 degrees away from what it was at the worst instant
```

That figure is measured by playing the reduced track back and comparing it to
the original at the same instants, and an independent sweep measures
16.321°. Two earlier versions of the metric were wrong and are described in
`docs/ifp_format.md`; both reported a large error for an animation that plays
back identically.

Neither is hidden, and neither is called an accuracy figure.

## Install

```
pip install -r requirements.txt          # converter and CLI
pip install -r requirements-dev.txt      # + the window and the test extras
```

The window needs PySide6; the command line does not, and works without it.

## The window

```
python run_gui.py
```

Six pages, in the order they are used.

| Page | What it does |
|---|---|
| **Input** | Pick the FBX and the DFF. Shows what was read: bone count, up axis, unit scale, clip list. |
| **Skeleton Mapping** | Every source bone, what it maps to, the IFP bone id from the DFF, and the confidence. Any row can be corrected by hand. |
| **Retarget Settings** | Root mode, FPS override, and opt-in key reduction. |
| **Preview** | Source and output side by side, same camera, scrubbed on one clock. |
| **Export** | Convert, write, read back, validate. |
| **Validation** | Per-bone max/mean/RMS angular error with the worst key named, and every diagnostic that fired. |

The preview draws the *ped's* skeleton on both sides, from the DFF's own bind
matrices. Drawing each rig in its own hierarchy would make a correct
conversion look wrong and an incorrect one look right, because the shapes
would not correspond at all.

## The command line

```
python cli.py inspect-fbx dance.fbx              # describe a source rig
python cli.py inspect-dff male01.dff            # describe a target ped
python cli.py check dance.fbx male01.dff        # can this be trusted?
python cli.py convert --fbx ... --dff ... --out ...
python cli.py doctor                            # what is installed, what works
```

`--json` on any command puts a single JSON document on stdout and the human
report on stderr, so it can be piped into a file and parsed.

Exit codes: `0` ok, `2` usage, `3` bad input, `5` validation failed.

Installed as a package it provides `gtafbx` and `gta-fbx-ifp` as commands.

### Options worth knowing

| Option | Default | Notes |
|---|---|---|
| `--block` | the clip name, sanitised | The animation name *inside* the IFP. `setPedAnimation` asks for this, not the file name. |
| `--root-motion` | off (in place) | Off by default: an IFP that walks the ped across the map is usually not what was wanted. |
| `--clip` | the first clip | When the FBX has several. |
| `--quiet`, `--no-color`, `--json` | | |

## How it works

Each stage below is a module with tests. The order is the order the work
happens in.

1. **FBX parse** — `fbx/`. Binary FBX, real animation curves, no third-party
   animation library. Detects handedness, up axis and unit scale from the
   file rather than assuming them, and applies a deterministic global
   conversion.
2. **DFF + HAnim** — `gta/dff_reader.py`. Reads the ped, resolves its
   skeleton, and produces the IFP bone id for every bone. **No id is ever
   inferred from a name**; a bone that does not resolve is reported, not
   guessed.
3. **Mapping** — `mapping/`. Semantic roles, aliases, hierarchy, direction,
   position and orientation scoring. Every match carries a confidence and a
   reason, and a mapping below the floor blocks the export until it is fixed.
4. **Rest pose** — `retarget/rest_pose.py`. Solves the correction between the
   two rigs' rest poses, including the root's orientation and each bone's
   local axis. Unit scale is measured from both skeletons rather than
   hard-coded — the Samba rig is 0.010766, not 0.01.
5. **Retarget** — `retarget/transfer.py`. Transfers rotation through the
   solved rest pose, preserving each limb's direction so elbows and knees
   bend the way they do on the source.
6. **IFP write** — `gta/ifp_writer.py`. ANP3. XYZW quaternions, `round(x *
   4096)`, clamped to int16, no gratuitous renormalisation. IFP names are 24
   bytes and an over-long one is **rejected**, never silently truncated.
7. **Validate** — `validate/`. Re-reads the file from disk and measures it.

Quaternions are XYZW throughout, matching the file format, so nothing is
swapped at the last moment.

## Checks that can fail

`validate/diagnostics.py` names the ways this can go wrong, because "the
output looks fine" is not a diagnostic:

- `mirrored_mapping` — one side driving the other's direction
- `swapped_sides` — a left bone mapped to a right one
- `reversed_joint` — an elbow or knee bending backwards
- `root_orientation` — the character stands upside down
- `invalid_quaternion` — a non-unit or NaN quaternion
- `axis_mismatch` — the ped's up axis does not point up, the Y-up/Z-up swap.
  Measured from the head bone's rotation against the DFF's own bind frame;
  the earlier version read the head's *translation*, and IFP stores
  translation on the root only, so that version could never fire
- `invalid_hanim_id` — an id that is not in the DFF
- `corrupt_anp3` — a header or frame that does not parse

Failures block the export. Warnings do not, and say what they are: a ped's
own bones are not mirror-symmetric — on `male01.dff` the two upper arms are
179° from mirrored — so that check warns rather than failing, or it would
fire on every correct conversion and mean nothing.

## What the tests found

Real bugs, each with the test that catches it:

- The right leg was silently unmapped, because GTA's tags mirror the way
  Mixamo's names do not.
- The rest-pose correction stood the character at 179°.
- A 26% timing stretch from the 1/50 s clock.
- `mean_quaternion` returned a fixed rotation regardless of its input: the
  outer product of two rotations is the identity, and it was passing a 3×3 to
  a function that decomposes a 4×4.
- `convert` was a stub that ran `check` and printed "not implemented", while
  the stage table listed six implemented stages as missing.
- `--json` printed a title line and the human report ahead of the JSON, so
  the output could not be parsed.
- The "drop negligible keys" checkbox on the settings page did nothing; no
  reduction code existed. A control that looks like it works is worse than
  none.
- Three separate diagnostics crashed on the errors they were written to
  report, because `GtaSkeleton` has `source_path` and no `.name`.
- A failed validation could print a custom headline and never the words
  `FAILED VALIDATION`, which is the phrase the user is told to look for.

```
$ pytest -q
188 passed
```

The resampling one is worth singling out: on a fast rotation the previous
keep-the-claiming-key behaviour was measurably wrong by 14°, it produced
byte-identical file sizes, and nothing about the file said so.

## Known limits

- **The 1/50 s time divisor is not independently verified** against a binary
  IFP from the game. It is consistent with the bundled fixtures and with
  published IFP documentation, but the shipped `examples/samba.ifp` has not
  been played in GTA itself. Everything else in this README is checked by
  the tests; this is the one claim that rests on the format documentation.
- Fingers, and any source bone with no IFP equivalent, are dropped and named
  in the report.
- GTA has no finger bones at all, so hand animation cannot survive the
  conversion however good the mapping is.
- `male01.dff` is the only ped the tests run against. The mapping is not
  hard-coded to it, but only it is exercised.

## Layout

```
cli.py                     the command line
run_gui.py                 starts the window
gta_fbx_ifp_converter/     the pipeline
    core/                  quaternions, space conversion
    fbx/                   FBX parsing, source rig
    gta/                   DFF, HAnim, IFP read/write
    mapping/               roles, aliases, the mapper
    retarget/              rest pose, transfer
    validate/              validator, diagnostics
gui/                       the window
    app.py                 the six pages
    pipeline.py            the only thing that touches the converter
    jobs.py                background threads
    preview.py             the side-by-side view
examples/                  the MTA resource and a generated IFP
tests/                     188 tests
testdata/                  a real FBX and a real DFF
docs/                      format notes
```

## Testing without a display

PySide6 links against `libGL`, `libEGL`, `libxkbcommon` and `libdbus-1`
whether or not it calls them. On a bare container none are installed and
`import PySide6` fails outright. `tests/qtstubs/` holds stand-ins that
satisfy the loader; `conftest.py` puts them on the path only when the real
import has already failed, so a normal desktop never loads them.

```
QT_QPA_PLATFORM=offscreen pytest tests/test_gui.py -q
```

## Trying it in MTA:SA

`examples/` has a complete, working resource. See
[`examples/README.md`](examples/README.md).

## Credits

FBX fixture: *Samba Dancing* from three.js's examples.
Ped: `male01.dff` from a GTA:SA skin.
