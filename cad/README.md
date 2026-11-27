# Pantsir-S1 radar (concept CAD)

Conceptual SolidCraft model of the **96K6 Pantsir-S1** radar fit on the turret in your reference photo (KAMAZ truck variant).

## What is modeled

| Part | Real system | Notes |
|------|-------------|--------|
| Front flat panel + frame | **1RS2-1** Ku-band engagement PESA | Panel size is **approximate** (open sources rarely publish aperture size). Tilt **18°** matches the photo. |
| Upper box | **2RL80** S-band search radar | **1776 × 940 mm** from published 2RL80 specs; depth is estimated. |
| Small roof pad | Turret interface | Placeholder only—not the full weapon turret. |

Units are **millimeters**. Bodies stay separate so you can move, fillet, or detail them in SolidCraft.

Reference image: `pantsir-s1-reference.png`

## Build locally (SolidCraft required)

From this folder (easiest):

```sh
./build.sh
```

Or manually:

```sh
solidcraft-cli run pantsir-s1-radar.json
```

Open the result:

```sh
solidcraft pantsir-s1-radar.sc
```

Or in one step:

```sh
solidcraft-cli run pantsir-s1-radar.json --save pantsir-s1-radar.sc
solidcraft pantsir-s1-radar.sc
```

Export for other tools:

```sh
solidcraft-cli run pantsir-s1-radar.json --out pantsir-s1-radar.step
```

## Tune dimensions

Edit the `parameters.change` block at the top of `pantsir-s1-radar.json`, or override from CLI:

```sh
solidcraft-cli run pantsir-s1-radar.json --param "engage_width=1600 mm" --param "engage_tilt=15 deg"
```

## Next steps in SolidCraft (local agent)

Use a **local** Cursor chat with SolidCraft MCP and ask to:

1. `open` this `.sc` file and `inspect_design` with `{ "measure": true }`.
2. Add **fillets** on the 2RL80 housing and a **shell** on the engagement housing.
3. Replace the search block with a **sketch + revolve** radome if you want a closer silhouette.
4. Attach only the radar group to a separate vehicle CAD via `file.insert_step`.

This cloud agent cannot run SolidCraft; all builds happen on your machine.