# IFP format notes

What this project assumes about the format, and which of it is verified.

## Layout (ANP3)

An IFP is a container of animations. Each is:

| Field | Size | Notes |
|---|---|---|
| `ANP3` | 4 | magic |
| version | 4 | 3 |
| `OffsetEOF` | 4 | **relative to byte 8**, so total size is `OffsetEOF + 8` |
| `TotalObjects` | 4 | number of frame groups in this animation |
| `FrameSize` | 4 | per-frame payload size, not the block size |
| `Name` | 24 | null-padded |
| `FrameCount` | 4 | keyframes per object |
| `BoneID` | 4 | the IFP bone id |
| `Flags` | 4 | bit 0 = compressed; bits 1-2 = frame type (3 = child, 4 = root) |

Frames follow the object header, `FrameCount` of them, at `FrameSize` bytes
each. The engine reads them **sequentially**; it does not navigate by
`OffsetEOF` or `FrameSize`. Anything that relies on those fields to find
data is relying on something the engine ignores.

## Quantisation

- rotation: four `int16`, **XYZW** order, scaled by 4096
- translation: three `int16` on the root, scaled by 1024

The writer uses `round(component * scale)`, then clamps to int16. Rounding
rather than truncating avoids the half-step bias toward zero that truncation
puts on every component of every key. Clamping rather than wrapping keeps an
out-of-range value from silently becoming a large rotation the other way.

No renormalisation happens after quantisation. The quaternion is already
unit-length, and dividing by a norm computed from the quantised values
changes the angles the file was supposed to store.

## Time

Frame times are 16-bit values in **1/50 s** units.

This is the one assumption in the project that is **not independently
verified** against a binary IFP shipped by the game. It is consistent with
the published IFP documentation and with the bundled fixtures, but nobody has
played `examples/samba.ifp` in GTA and confirmed the speed.

The consequence is unavoidable: a 60 fps source produces 1149 keys per track
where the format can store 557, and the ones that share a slot must be
merged. The tool reports how many were lost rather than presenting the
reduced key count as the source's.

## What was checked against real data

- Header sizes and field order: `male01.dff` and a generated IFP re-read by
  this project's own strict parser
- Bone ids: every one resolves through `Target DFF → HAnim → effective bone id`;
  the round-trip test asserts no id appears in the output that is not in the
  DFF
- Round-trip error: the written file is re-read from disk and compared against
  the data the writer was given. The Samba clip comes back at 0.0265°
  max angular error and 0.08 mm max position error across 13,925 keys.

That last figure is **the cost of the format**, not conversion accuracy. The
distinction is stated in the README too, because a number that looks small
and means something other than what a reader would assume is worse than no
number at all.

## Sources

- <https://gtamods.com/wiki/IFP> — header layout, name sizes, frame types,
  scales
- <https://github.com/multitheftauto/mtasa-blue/blob/master/Client/mods/deathmatch/logic/CClientIFP.h>
  — the structures the engine actually uses
- <https://gtaforums.com/topic/400901-creating-custom-ifps/> — `OffsetEOF` is
  relative to byte 8
