# Reference dumps

Ground-truth data used by the test suite. Do **not** hand-edit these files.

## `threejs_local_rotations.json`

Per-bone local rotations for `../samba_dancing.fbx`, produced by
[three.js](https://threejs.org/) `FBXLoader` (three@0.160.0), the reference
FBX importer — and the renderer for this exact fixture, which ships it as
`examples/models/fbx/Samba Dancing.fbx`.

Each entry stores every 10th key of a bone's `.quaternion` animation track:
the track's own key time in seconds and the quaternion as
`[x, y, z, w]`. Times and quaternions are written with their full float
precision; rounding them makes the comparison in
`tests/test_fbx_source_rig.py::test_local_rotations_match_the_reference_loader`
fail on keys that land a few microseconds either side of a real key.

### Regenerating

```bash
mkdir -p /tmp/fbxref && cd /tmp/fbxref
npm init -y >/dev/null && npm install three@0.160.0
```

```js
// dump.mjs
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import fs from 'fs';

const nb = fs.readFileSync('/path/to/samba_dancing.fbx');
// three.js compares the first 22 bytes, so hand it a plain ArrayBuffer
// rather than a Node Buffer, whose byteOffset would break the check.
const grp = new FBXLoader().parse(
  nb.buffer.slice(nb.byteOffset, nb.byteOffset + nb.byteLength), '');
grp.updateMatrixWorld(true);

const out = {};
for (const track of grp.animations[0].tracks) {
  if (!track.name.endsWith('.quaternion')) continue;
  const name = track.name.split('.')[0].replace('mixamorig', 'mixamorig:');
  const times = [], quaternions = [];
  for (let i = 0; i < track.values.length / 4; i += 10) {
    times.push(track.times[i]);
    quaternions.push([0, 1, 2, 3].map(k => track.values[i * 4 + k]));
  }
  out[name] = { times, quaternions };
}
const sorted = {};
for (const k of Object.keys(out).sort()) sorted[k] = out[k];
fs.writeFileSync('threejs_local_rotations.json', JSON.stringify(sorted));
```

```bash
node dump.mjs
```

### What this pins down

Three things in the FBX reader are only checkable against an independent
implementation, because each one produces output that looks plausible:

- **Euler angles are degrees.** This file stores rotations up to 765°.
  Read as radians the animation folds the skeleton through itself.
- **The default Euler order is `ZYX`, not `XYZ`.** No `RotationOrder`
  property exists in this FBX, so the reader falls back to the SDK default.
- **The order letters index the array by axis name, not by position.**
  Pairing them positionally swaps X and Z for every non-`XYZ` rig.

The current agreement is a maximum of `4.5e-6` degrees over 2869 samples,
which is bounded by three.js storing its track values as float32.
