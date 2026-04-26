# SCSP Fogpiercer

## Track: Wargaming

Local hackathon demo for a national defense AI competition.

## Team

- Beau Warren

## Data Source

Primary dataset: [CDB90 data](https://github.com/jrnold/CDB90/tree/master/data)
from the public `jrnold/CDB90` repository. The project uses this source for the
Logit Hierarchical Regression training table and maps live simulation sensor
features back into CDB90-shaped columns.

The project has two
parts:

- `logit_hierarchical_regression/`: training, local inference tests, feature
  schema, data staging, and final trained model artifacts.
- `simulation/`: local 2D battlefield simulation and UI that pulls model
  outputs from the trained model endpoint/artifact.

## Quick Start: Run the Demo Locally

These commands assume Ubuntu/WSL with Python 3.11 or newer.

1. Clone the repo and enter the project folder:

```bash
git clone https://github.com/fermsi-paradox/FogPiercer.git
cd FogPiercer
```

2. Create your local environment file:

```bash
cp .env.example .env
```

Open `.env` and fill in the API keys/model paths listed below. Do not commit
`.env`.

3. Install the backend Python requirements:

```bash
cd simulation/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Start the backend API:

```bash
PYTHONPATH="$PWD" uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep that terminal running. The API is now available at
`http://127.0.0.1:8000`.

5. In a second terminal, start the frontend:

```bash
cd FogPiercer/simulation/frontend
python3 -m http.server 5173
```

6. Open the demo in your browser:

```text
http://127.0.0.1:5173/
```

The frontend is a simple static page. The backend does the model loading,
decision scoring, telemetry logging, and Mercury II summaries.

## Optional: Train the Logit Model

The basic demo uses the trained model path configured in `.env`. To work on
model training instead, install the separate training requirements:

```bash
cd FogPiercer/logit_hierarchical_regression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training expects a prepared CDB90-style table at `CDB90_DATA_PATH` and a
Hugging Face repo/token configured in `.env`.

## API Keys and Secrets

This repo is intended to be safe to publish later. Do not commit `.env`, API
keys, raw restricted datasets, generated model binaries, or Hugging Face tokens.
Use `.env.example` as the source of truth for variable names.

Required local environment variables:

- `MERCURY_II_API_KEY`: API key for Inception Mercury II.
- `MERCURY_II_BASE_URL`: Mercury II API base URL, if different from the
  default configured later.
- `HF_TOKEN`: Hugging Face token with write access for training pushes and read
  access for demo pulls.
- `FOGPIERCER_HF_MODEL_REPO`: Hugging Face model repository, for example
  `your-hf-username/SCSP_fogpiercer-logit-hierarchical-regression`.
- `GITHUB_TOKEN`: optional, only for later GitHub automation.
- `CDB90_DATA_PATH`: local path to the CDB90 CSV.

## Step Checkpoints

1. Train and validate the Logit Hierarchical Regression model on local CDB90
   data, then push the trained artifact to Hugging Face.
2. Build the simple 2D visual environment with realistic unit symbology and
   drag-and-drop unit movement.
3. Generate synthetic sensor data from live simulation state using the same
   feature schema as the CDB90 training data.
4. Ask Mercury II to summarize the top 3 model-ranked decisions in plain
   operator-friendly language.
5. End/reset the demo when the VIP is killed, the friendly force is eliminated,
   the enemy force is eliminated, the clock expires, or the VIP reaches the
   exfil side of the map.

## Mission End Conditions

The frontend owns the live mission-ending logic because it has the current
unit positions every animation tick. The backend records the final status in
telemetry when the frontend logs the `ended`, `reset`, or `reiterate` event.

Current end states:

```text
if VIP health <= 0:
  mission_status = enemy_victory

if live VIP x-position >= 900:
  mission_status = friendly_extraction

if live friendly count == 0:
  mission_status = enemy_victory

if live enemy count == 0:
  mission_status = friendly_victory

if demo clock <= 0:
  mission_status = clock_expired
```

`friendly_extraction` means the VIP/convoy has made it through the ambush and
cleared the far/east side of the map. This is different from destroying every
enemy. It lets a heavily outnumbered friendly force still succeed by moving the
VIP through the road segment or surviving long enough to break contact.

The final UI and `simulation/logs/<run_id>/final_summary.json` include:

- `vip_killed`
- `vip_extracted`
- `all_friendlies_eliminated`
- `all_enemies_eliminated`
- friendly/enemy alive counts

## Simulation Controls and Tactical Logic

The left panel lets the user set friendly and enemy unit counts before or during
a reset. Existing units can be dragged directly on the map during a fight; the
next backend decision refresh uses the full live unit list and positions.

`Drone Ambush` is an optional scenario modifier next to the force-count inputs.
When checked, all enemy units spawn as FPV drones except one infantry unit
labeled `CTRL`, representing the drone controller. If `CTRL` is killed, all
enemy FPV drones are immediately eliminated. This changes only the scenario
composition and drone lifecycle; it does not change the Logit model, combat
damage values, or probability math.

Ground units cannot move through the building-wall rectangles defined in the
frontend, and building walls block ground-unit direct fire. FPV drones can fly
over buildings. FPV detonation still uses the configured FPV range, but the
frontend checks the drone's movement segment during each tick so a fast drone
does not visually cross a target and miss the impact due to frame timing.

Friendly units display faint dotted yellow sensor rings. These rings are visual
only, but their radius matches the backend local contact range used by the
contact-pressure multiplier.

## Decision Percentage Math

The displayed decision percentage is a calibrated probability. It starts with
the trained Logit Hierarchical Regression model, then applies live battlefield
multipliers so the UI does not show impossible-looking outcomes such as one VIP
vehicle defeating eight hostile units with near certainty.

### 1. Build One Candidate Row Per Action

For each available action, the backend creates one CDB90-shaped feature row:

```text
candidate_features[action] = base_battlefield_features + action_feature_deltas
```

`base_battlefield_features` comes from live simulation state through
`simulation/backend/app/sensor_mapping.py`. Examples include friendly/enemy
weighted combat power, alive counts, VIP health, hostile FPV drone presence, and
enemy proximity.

`action_feature_deltas` comes from
`simulation/backend/app/candidate_actions.py`. For example, counter-drone action improves
`techa`, `intela`, `aeroa`, and `aira`, while break-contact improves `resa` and
changes `direction`.

### 2. Preprocess Features

The trained pipeline preprocesses the feature row exactly as it was trained:

```text
numeric_feature = median_impute(numeric_feature)
numeric_feature_scaled = (numeric_feature - training_mean) / training_std

categorical_feature = mode_impute(categorical_feature)
categorical_feature_encoded = one_hot_encode(categorical_feature)
```

After preprocessing, the row is transformed into a numeric vector:

```text
phi(x) = preprocessed numeric and one-hot encoded feature vector
```

### 3. Raw Logit Probability

The Logistic Regression model estimates the probability of `attacker_success`:

```text
logit = beta_0 + beta_1 * phi_1(x) + beta_2 * phi_2(x) + ... + beta_n * phi_n(x)

raw_logit_probability = sigmoid(logit)

sigmoid(logit) = 1 / (1 + exp(-logit))
```

In code this is:

```text
raw_logit_probability = pipeline.predict_proba(candidate_frame)[:, 1]
```

This is the historical CDB90-like prior. It answers: given a row shaped like
CDB90, how often did similar attacker/action conditions succeed historically?

### 4. Live Force-Balance Multiplier

The simulation then calculates live weighted combat power:

```text
unit_weight:
  VIP  = 0.35
  UAS  = 0.65
  INF  = 1.00
  MRAP = 1.45

friendly_weighted_power = sum(unit_health * unit_weight for live friendly units)
enemy_weighted_power    = sum(unit_health * unit_weight for live enemy units)

power_ratio = friendly_weighted_power / enemy_weighted_power
count_ratio = friendly_alive_count / enemy_alive_count

blended_ratio = (0.75 * power_ratio) + (0.25 * count_ratio)

force_balance_multiplier = clamp(blended_ratio ** 0.45, 0.05, 1.15)
```

Special cases:

```text
if enemy_weighted_power <= 0: force_balance_multiplier = 1.15
if friendly_weighted_power <= 0: force_balance_multiplier = 0.05
```

This multiplier is why `1 VIP vs 8 enemies` now drops sharply even when the raw
historical logit probability is high.

### 5. Action-Fit Multiplier

Each action also receives an action-fit multiplier:

```text
counter-drone:
  1.00 if an enemy FPV drone is alive
  0.10 if no enemy FPV drone is alive

screen-and-push:
  0.45 if enemy_count >= max(2, friendly_count * 2)
  0.90 if friendly_weighted_power >= enemy_weighted_power * 0.8
  0.55 otherwise

break-contact:
  0.95 if enemy_weighted_power > friendly_weighted_power or VIP health < 70
  0.75 otherwise

shift-vip-to-cover:
  0.90 if VIP health < 80 or enemy_weighted_power > friendly_weighted_power
  0.65 otherwise

hold-defensive-perimeter:
  0.85 if enemy_count > friendly_count
  0.55 otherwise

call-for-reinforcement:
  0.80 if enemy_weighted_power > friendly_weighted_power * 1.25
  0.45 otherwise
```

### 6. Displayed Probability

The final displayed probability is:

```text
adjusted_probability =
  raw_logit_probability
  * force_balance_multiplier
  * action_fit_multiplier
  * contact_pressure_multiplier

displayed_probability = clamp(adjusted_probability, 0.01, 0.99)
displayed_percent = round(displayed_probability * 100, 1)
```

The UI sorts all candidate actions by `displayed_probability`. The top three are
shown as prominent selectable options. Lower-ranked options are still visible,
but they are smaller gray cards.

The raw data popup shows the formula inputs:

- `raw_logit_probability`
- `force_balance_multiplier`
- `action_fit_multiplier`
- `contact_pressure_multiplier`
- `probability_formula`
- local sensor/contact counts and local pressure power

### 7. Selected Decision Trend

The active-decision header shows the trend for the currently selected option
only. This is a frontend decision-support signal, not an additional Logit model
feature.

When the user selects a top-ranked decision, the frontend stores:

```text
selected_trend = {
  id: selected_decision_id,
  score: selected_displayed_percent,
  at: current_timestamp_ms,
  slope: 0
}
```

On each decision refresh, the frontend finds the latest score for that same
decision id. If the score changed by at least `0.05` percentage points, it
calculates the current slope:

```text
elapsed_minutes = max((now_ms - previous_timestamp_ms) / 60000, 1 / 60)

slope_pp_per_minute =
  clamp(
    (current_displayed_percent - previous_displayed_percent)
    / elapsed_minutes,
    -99,
    99
  )
```

The UI displays this as `Selected trend: +X.X pp/min`, `Selected trend: -X.X
pp/min`, or `Selected trend: steady` when the absolute slope is below `0.25`.
The trend resets when the commander selects a different option or resets the
scenario.

### 8. Local Contact-Pressure Multiplier

The backend also applies a local contact-pressure multiplier. This separates
global force balance from immediate local danger, especially in FPV-heavy drone
ambushes where drones have low health but high one-shot lethality.

The local sensor/contact radius is:

```text
local_sensor_range = 220 px ~= 550 m
```

The frontend draws the same radius as faint dotted yellow sensor rings around
live friendly units.

This would let a `3 vs 8` scenario remain dangerous overall while still showing
better odds if only one enemy is inside firing/proximity range and the three
friendlies can focus on that enemy. Conversely, if several enemies are close
enough to engage the VIP or friendly units at once, the multiplier would reduce
the displayed probability sharply.

```text
local_enemies =
  enemies within local_sensor_range of any live friendly unit

local_friendlies =
  friendlies within local_sensor_range of those local enemies

local_enemy_pressure_power =
  sum(max(enemy_health * unit_weight, 95) for local enemy FPV drones)
  + sum(enemy_health * unit_weight for other local enemies)

local_ratio =
  local_friendly_weighted_power / max(1, local_enemy_pressure_power)

local_balance =
  clamp(local_ratio ** 0.35, 0.35, 1.10)

fpv_salvo_penalty =
  max(0.52, 1 - (0.12 * local_enemy_drone_count))

contact_pressure_multiplier =
  clamp(local_balance * fpv_salvo_penalty, 0.25, 1.10)
```

This means one nearby FPV creates a modest penalty, while several nearby FPVs
create a strong penalty even though each drone has low health. If the drone
controller is killed in the `Drone Ambush` mode and enemy FPVs are eliminated,
the local FPV pressure disappears on the next backend decision refresh.

## Logit Feature Columns

The trained target is `attacker_success`. The action column is
`tactical_posture`. The grouped context columns are `war4_theater` and
`terrain_primary`.

| Feature | Type | Explanation |
| --- | --- | --- |
| `tactical_posture` | Categorical | Candidate action posture label sent to the model, such as `HD\|postype_0`, `PD\|postype_0`, or `PD\|postype_1`. |
| `war4_theater` | Categorical | Broad conflict/theater grouping. The demo uses `Modern local demo`. |
| `terrain_primary` | Categorical | Primary terrain category for grouping and model context. The demo uses `R` for road/restricted route terrain. |
| `weather_primary` | Categorical | Primary weather condition. The demo uses `D` as the dry/default condition. |
| `primary_attacker` | Categorical | CDB90-style attacker actor. In the demo this is the friendly VIP escort because the options are framed as friendly actions. |
| `primary_defender` | Categorical | CDB90-style defender actor. In the demo this is the hostile ambush cell. |
| `postype` | Numeric | Numeric posture type from CDB90-derived tactical posture. `0` is a single-posture action, `1` is a combined posture. |
| `post1` | Categorical | First tactical posture code. Examples include `HD` for hasty defense and `PD` for prepared defense/withdrawal-style posture. |
| `post2` | Categorical | Optional second tactical posture code for combined actions. Often `None` for single-posture actions. |
| `front` | Numeric | Whether the fight is treated as active front/contact. Set when enemy pressure or close contact is high. |
| `depth` | Numeric | Whether the fight has depth/penetration pressure. Set when enemy power or close contact implies deeper threat. |
| `aeroa` | Numeric | Attacker air/aerial advantage. Hostile FPV drones can reduce this; counter-drone action can improve it. |
| `surpa` | Numeric | Attacker surprise advantage. Close enemy contact or overmatch reduces friendly surprise. |
| `cea` | Numeric | Combat effectiveness/force-balance proxy. Built from weighted friendly/enemy combat power and count conditions. |
| `leada` | Numeric | Attacker leadership/command signal. Enabled when a commander decision is active. |
| `trnga` | Numeric | Attacker training advantage. Currently neutral in the demo. |
| `morala` | Numeric | Attacker morale/cohesion proxy. Improved by friendly overmatch or healthy VIP, reduced by overmatch against friendlies or VIP damage. |
| `logsa` | Numeric | Attacker logistics/support signal. Improved by call-for-reinforcement. |
| `momnta` | Numeric | Attacker momentum. Improved by push/overmatch, reduced by withdrawal, defensive hold, or being badly outnumbered. |
| `intela` | Numeric | Attacker intelligence/air-awareness advantage. Friendly FPV availability and counter-drone action improve this; hostile FPV drones can reduce it. |
| `techa` | Numeric | Attacker technology/drone advantage. Friendly FPV availability and counter-drone action improve this. |
| `inita` | Numeric | Attacker initiative. Screen-and-push and friendly power advantage improve this; break-contact and reinforcement delay reduce it. |
| `quala` | Numeric | Attacker force quality/survivability proxy. In the demo it is strongly influenced by VIP health and shift-to-cover actions. |
| `resa` | Numeric | Attacker reserve/resilience/recovery signal. Break-contact and VIP-to-cover increase this. |
| `mobila` | Numeric | Attacker mobility. Screen-and-push and shift-to-cover increase this; break-contact and static defense reduce it. |
| `aira` | Numeric | Attacker air operations/air control signal. Hostile FPV drones reduce this; counter-drone action improves it. |
| `fprepa` | Numeric | Fire preparation/preparatory fires signal. Currently neutral in the demo. |
| `wxa` | Numeric | Weather advantage signal. Currently neutral in the demo. |
| `terra` | Numeric | Terrain advantage signal. The demo sets this to `1` for restricted road/cover terrain. |
| `leadaa` | Numeric | Alternate leadership/command feature from CDB90. Enabled when a commander decision is active. |
| `plana` | Numeric | Planning/coordination advantage. Improved by active command, screen-and-push, defensive hold, or VIP-to-cover. |
| `surpaa` | Numeric | Alternate surprise feature. Currently neutral unless future mappings use it. |
| `mana` | Numeric | Manpower/management/support feature. Improved by call-for-reinforcement. |
| `logsaa` | Numeric | Alternate logistics feature. Improved by call-for-reinforcement. |
| `fortsa` | Numeric | Fortification/defensive posture feature. Improved by hold-defensive-perimeter. |
| `deepa` | Numeric | Deep fight/depth pressure signal. Set when enemy overmatch or close contact creates depth pressure. |
| `is_hero` | Numeric | CDB90 hero/actor-side indicator. The demo fixes this to `1` for the friendly action side. |
| `war_initiator` | Numeric | Whether the modeled side is the initiator. The demo fixes this to `1` for friendly candidate actions. |
| `terrano` | Numeric | Number/count flag for terrain categories. The demo uses `1`. |
| `terra1` | Categorical | First terrain code. The demo uses `R` for road/restricted terrain. |
| `terra2` | Categorical | Second terrain code. The demo uses `M` for mixed terrain context. |
| `terra3` | Categorical | Third terrain code. Usually `None` in the demo unless a mapping supplies another terrain context. |
| `wxno` | Numeric | Number/count flag for weather categories. The demo uses `1`. |
| `wx1` | Categorical | First weather code. The demo uses `D`. |
| `wx2` | Categorical | Second weather code. The demo uses `S` as a static CDB90-compatible weather category. |
| `wx3` | Categorical | Third weather code. The demo uses `T` as a static CDB90-compatible weather category. |
| `wx4` | Categorical | Fourth weather code. The demo uses `F` as a static CDB90-compatible weather category. |
| `wx5` | Categorical | Fifth weather code. The demo uses `T` as a static CDB90-compatible weather category. |
| `duration1` | Numeric | Baseline duration feature. The demo uses `1.0`. |
| `duration2` | Numeric | Elapsed scenario time converted to days: `(300 - seconds_remaining) / 86400`. |
| `dyad_weight` | Numeric | CDB90 dyad/sample weight. The demo uses `1.0`. |
| `direction` | Numeric | Direction of action. `1` is forward/continue, `-1` is reverse/break-contact. |

