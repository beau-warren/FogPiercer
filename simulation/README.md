# Simulation

This folder will contain the local 2D wartime scenario demo:

- A lightweight browser UI with a top-level situation banner, current selected
  decision, and always-visible top 3 recommended decisions.
- A 2D map/canvas with simple military-style icons for friendly vehicles,
  infantry, drones, hostile drones/infantry, road, cover, and sensor zones.
- Drag-and-drop unit manipulation so the model updates in real time.
- A local backend that turns live simulation state into CDB90-shaped sensor
  features, calls the Logit Hierarchical Regression model, and asks Mercury II
  to summarize the top decisions.
- Reset/reiterate controls for a 5-minute demo loop.

No API keys should be stored here. The backend will read the root `.env`.

## Step 2 Frontend

The first simulation pass is a dependency-free static app in `frontend/`.

Run it locally:

```bash
cd /home/fermsi/github_repos/SCSP_fogpiercer/simulation/frontend
python3 -m http.server 5173
```

Then open:

```text
http://localhost:5173
```

Implemented in this pass:

- Randomized 2D battlefield with road, cover, danger zone, and sensor rings.
- Simple military-style symbology: blue friendly rectangles/circles and red
  hostile diamonds.
- Friendly VIP convoy, escort, infantry, friendly ISR drone, hostile drone, and
  hostile infantry cells.
- Coordinated enemy movement toward the convoy.
- Drag-and-drop movement for any live unit.
- Always-visible top 3 placeholder decisions that update with battlefield
  state.
- Selected decision banner at the top and unit behavior changes after selection.
- End state when one side is neutralized or the 5-minute demo clock expires.
- Reset/reiterate controls.

The current decision scores are local heuristics. Step 3 will replace those
scores with CDB90-shaped synthetic sensor features passed into the trained Logit
Hierarchical Regression model.

