# DFF, HAnim, and how a bone gets its id

The rule this project is built around:

> **Every bone id in the output must resolve through
> `Target DFF → HAnim → effective bone id`. Nothing is inferred from a name.**

## Why the rule exists

An IFP frame names a bone by a small integer. That integer is an index into
the ped's animation skeleton, and it is *not* the same as:

- the order the bones appear in the model's hierarchy
- the HAnim node id stored in the file
- the "canonical tag" (`SaBoneTag`) that identifies which bone it is

Getting this wrong produces a file that loads, plays, and moves the wrong
limb. Nothing errors. That is the failure mode worth engineering against.

## The three numbers

For `male01.dff`, the right upper arm:

| | value | where it comes from |
|---|---|---|
| HAnim node id | 301 | the raw `node_id` in the DFF's HAnim section |
| canonical tag | 22 | which bone it is (`R_UPPERARM`) |
| **IFP bone id** | **22** | the tag, resolved through the game's tag table |

`R Toe0` is the interesting one, and it is why the raw node id is not enough:

| | value |
|---|---|
| HAnim node id | **absent** |
| canonical tag | 54 (`R_TOE0`) |
| **IFP bone id** | **54** |

The HAnim node id is missing, and the bone is still perfectly animatable,
because the id comes from the tag. An earlier version of the mapping table
displayed the HAnim node id as *the* animation id and showed this bone as
having none. The table now shows the resolved id and keeps the node id in a
tooltip.

## Resolution

`gta/tag_resolve.py` maps each bone's canonical tag to an IFP id. A bone
whose tag is unknown gets **no id** and is reported as unresolvable. It is
never given a fallback, a neighbour's id, or an index into the hierarchy.

`gta/dff_reader.py` exposes `GtaBone.bone_id` (what gets written),
`GtaBone.hanim_id` (the raw node id, for diagnostics) and
`GtaBone.is_addressable` (whether it can carry a track at all).

The test suite asserts that every id in a written file is present in the DFF
that was read, so an invented id fails the build rather than the player's
screen.

## Bones with no IFP equivalent

A ped has no fingers. A Mixamo rig has 24 of them, and so does the target
skeleton have nothing to map them to. They are named in the report:

```
42 source bones have no IFP representation; see the report for the full list
```

This is a property of the target, not a failure. It is reported so it is
never mistaken for motion that was silently lost.
