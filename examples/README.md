# MTA:SA example

`samp_anim_play.resource` loads the IFP next to it and plays it on a ped.

## Install

1. Put both files in your MTA `server-resources/` directory:

       server-resources/
         samp_anim_play.resource
         samba.ifp

2. Add the resource to `server.cfg`:

       addResource "samp_anim_play.resource"

3. Start the server. The log should say:

       [animplay] samba.ifp contains: SAMBA
       [animplay] ready -- /animplay near a ped

4. Walk up to any ped in-game and type `/animplay`.

## Regenerating the IFP

`examples/samba.ifp` was produced from the bundled test FBX and DFF:

    python cli.py convert \
        --fbx testdata/samba_dancing.fbx \
        --dff testdata/male01.dff \
        --out examples/samba.ifp \
        --block SAMBA

The `--block` name is what the resource asks `engineLoadIFP` for, and the
test suite checks that the two still agree — they are two files referring to
each other by name, and nothing else would notice if you edited one.

## What it checks, and why

A file can pass every check in the converter and still be one the game will
not animate. Only the game settles that, so the resource asks the game:

- `engineLoadIFP` returning a non-empty table means the file parsed
- walking that table and logging the names means a wrong block name is
  visible rather than silent — a name that is not in the file is simply
  absent, not an error
- `setPedAnimation` returning true means the engine accepted the track
- `getElementAnimationState` half a second later is the game's own answer to
  "is this ped actually animating", which is the one that matters

If the last check fails while the ped is alive and in range, the IFP loaded
and did not play. That is a real defect and the server log says so.
