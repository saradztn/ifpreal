--[[
    samp_anim_play.resource

    Plays an animation this project converted from an FBX, using the IFP it
    wrote.  It is the test that matters: a file can pass every check in the
    converter and still be one the game refuses to load, or plays as a
    motionless ped, and only the game settles that.

    What it does, in order:
      1. loads the IFP with engineLoadIFP and checks it returned a real table
      2. finds an animation inside it that actually exists
      3. loops setPedAnimation on a ped so the animation can be seen moving
      4. reports, server side, whether the ped's animation state actually
         changed -- the game's own answer, not the converter's

    Install:
      put this file in server-resources/
      put the .ifp next to it
      add    addResource "samp_anim_play.resource"   to server.cfg
      start the server, then in-game /animplay

    engineLoadIFP keys the table it returns by the animation name stored
    *inside* the file, which is the --block given at conversion time and not
    necessarily the file name.  A name that is not present is simply absent
    rather than an error, so the loop below logs every name it found; a
    mismatch should be visible rather than silent.
]]

local ANIM_NAME  = "SAMBA"       -- the block name given to the converter
local IFP_FILE   = "samba.ifp"   -- the file next to this resource
local LOOP       = true         -- false plays it once
local SPEED      = 1.0
local RADIUS     = 2.0          -- how close the player has to be

local anims = engineLoadIFP(IFP_FILE)

if type(anims) ~= "table" or next(anims) == nil then
    outputServerLog("[animplay] engineLoadIFP returned nothing for " .. IFP_FILE ..
        " -- the file is missing, truncated, or not an IFP the game accepts")
    outputChatBox("[animplay] could not load " .. IFP_FILE)
    return
end

local found = nil
local listed = {}
for name in pairs(anims) do
    listed[#listed + 1] = name
    if name == ANIM_NAME then
        found = true
    end
end
table.sort(listed)
outputServerLog("[animplay] " .. IFP_FILE .. " contains: " .. table.concat(listed, ", "))

if not found then
    outputServerLog("[animplay] '" .. ANIM_NAME .. "' is not in " .. IFP_FILE ..
        ".  Rebuild it with:")
    outputServerLog("[animplay]   python cli.py convert --fbx dance.fbx" ..
        " --dff male01.dff --out " .. IFP_FILE .. " --block " .. ANIM_NAME)
    outputChatBox("[animplay] '" .. ANIM_NAME .. "' is not in that IFP")
    return
end

local function pedDistance(ped)
    local player = getLocalPlayer()
    if not player or not getElementByID(player) then
        return math.huge
    end
    return getDistanceBetweenPointsAnd3D(
        getElementPosition(player),
        getElementPosition(ped))
end

local function playFor(ped)
    -- A ped that is dead, in a car, or is the player cannot be made to play a
    -- custom animation; the call returns false and the reason is the game's,
    -- so it is logged rather than guessed at here.
    if not getElementByID(ped) then return false, "no such ped" end
    if getElementType(ped) ~= "ped" then return false, "not a ped" end
    if isPedDead(ped) then return false, "the ped is dead" end
    if getPedInVehicle(ped) then return false, "the ped is in a vehicle" end

    if not setPedAnimation(ped, IFP_FILE, ANIM_NAME, SPEED, LOOP, false, false) then
        return false, "setPedAnimation refused it"
    end
    return true, nil
end

local function animplay(player)
    local target = getElementByID(player)
    if not target then return end

    local peds = getElementsByType("ped", false)
    local chosen, refused = nil, nil
    for i = 1, #peds do
        if isElement(peds[i]) and pedDistance(peds[i]) < RADIUS and peds[i] ~= target then
            local ok, why = playFor(peds[i])
            if ok then
                chosen = peds[i]
                break
            else
                refused = why
            end
        end
    end

    if not chosen then
        outputChatBox("[animplay] " .. (refused or
            ("no ped within " .. RADIUS .. "m -- walk up to one first")))
        return
    end

    -- The game's own answer, read back a moment later.  A converter that
    -- wrote a file the engine loaded but cannot animate shows up here as
    -- an unchanged animation state.
    setTimer(function()
        if not isElement(chosen) then return end
        if not getElementAnimationState(chosen) then
            outputServerLog("[animplay] the ped is alive and inside range but" ..
                " reports no animation state -- " .. ANIM_NAME ..
                " loaded and did not play")
            outputChatBox("[animplay] loaded, but it is not playing")
            return
        end
        outputChatBox("[animplay] playing " .. ANIM_NAME)
        outputServerLog("[animplay] playing " .. ANIM_NAME .. " on ped " ..
            tostring(getElementID(chosen)))
    end, 500, 1)
end

addEventHandler("onResourceStart", resourceRoot)
addCommandHandler("animplay", animplay, false, false)
addEventHandler("onPlayerSpawn", animplay)

outputServerLog("[animplay] ready -- /animplay near a ped")
