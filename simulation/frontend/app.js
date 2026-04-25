const WIDTH = 1000;
const HEIGHT = 620;
const TICK_MS = 240;
const DEMO_SECONDS = 5 * 60;
const API_BASE_URL = "http://127.0.0.1:8000";

const svg = document.querySelector("#battlefield");
const terrainLayer = document.querySelector("#terrain-layer");
const sensorLayer = document.querySelector("#sensor-layer");
const movementLayer = document.querySelector("#movement-layer");
const fireLayer = document.querySelector("#fire-layer");
const unitLayer = document.querySelector("#unit-layer");
const decisionList = document.querySelector("#decision-list");
const sensorSummary = document.querySelector("#sensor-summary");
const situationTitle = document.querySelector("#situation-title");
const situationCopy = document.querySelector("#situation-copy");
const currentDecision = document.querySelector("#current-decision");
const battlefieldStatus = document.querySelector("#battlefield-status");
const demoClock = document.querySelector("#demo-clock");
const resetButton = document.querySelector("#reset-button");
const reiterateButton = document.querySelector("#reiterate-button");
const rawDataPopover = document.querySelector("#raw-data-popover");
const rawDataBody = document.querySelector("#raw-data-body");
const rawDataTitle = document.querySelector("#raw-data-title");
const rawDataClose = document.querySelector("#raw-data-close");
const rawDataDragHandle = document.querySelector("#raw-data-drag-handle");

const terrainTemplates = [
  {
    road: "M 80 490 C 240 420, 330 380, 500 330 S 790 235, 930 140",
    cover: [
      [90, 95, 150, 95],
      [735, 345, 170, 110],
      [545, 85, 150, 95],
    ],
    danger: [595, 225, 245, 165],
  },
  {
    road: "M 65 420 C 230 350, 345 365, 500 310 S 740 215, 930 230",
    cover: [
      [125, 115, 160, 105],
      [680, 405, 185, 100],
      [480, 95, 150, 105],
    ],
    danger: [560, 250, 280, 150],
  },
  {
    road: "M 75 520 C 205 470, 330 415, 450 420 S 660 365, 905 120",
    cover: [
      [75, 125, 185, 115],
      [590, 125, 150, 95],
      [745, 395, 165, 105],
    ],
    danger: [500, 300, 275, 145],
  },
];

const state = {
  runId: createRunId(),
  tick: 0,
  units: [],
  terrain: terrainTemplates[0],
  selectedDecision: null,
  secondsRemaining: DEMO_SECONDS,
  ended: false,
  dragging: null,
  tickHandle: null,
  lastDecisionRenderAt: 0,
  decisionSignature: "",
  renderedDecisions: [],
  fireTracers: [],
  popoverDrag: null,
  modelDecisions: [],
  modelOnline: false,
  modelFetchInFlight: false,
  lastModelFetchAt: 0,
  endLogged: false,
};

function randomBetween(min, max) {
  return min + Math.random() * (max - min);
}

function createRunId() {
  return `run-${new Date().toISOString().replace(/[:.]/g, "-")}-${Math.random()
    .toString(16)
    .slice(2, 8)}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function nearestUnit(source, candidates) {
  return candidates
    .filter((unit) => unit.health > 0)
    .sort((a, b) => distance(source, a) - distance(source, b))[0];
}

function emitFireTracer(source, target) {
  state.fireTracers.push({
    id: `${source.id}-${target.id}-${Date.now()}-${Math.random()}`,
    x1: source.x,
    y1: source.y,
    x2: target.x,
    y2: target.y,
    expiresAt: Date.now() + 340,
  });
}

function createUnit({ id, side, type, label, x, y, speed, range, health = 100 }) {
  return {
    id,
    side,
    type,
    label,
    x,
    y,
    speed,
    range,
    health,
    targetX: x,
    targetY: y,
    lastX: x,
    lastY: y,
  };
}

function missionStatus() {
  const friendlyAlive = getFriendlyUnits().length;
  const enemyAlive = getEnemyUnits().length;
  if (friendlyAlive === 0) return "enemy_victory";
  if (enemyAlive === 0) return "friendly_victory";
  if (state.secondsRemaining <= 0) return "clock_expired";
  return "running";
}

function logSimulationEvent(eventType, status = missionStatus()) {
  fetch(`${API_BASE_URL}/api/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      runId: state.runId,
      eventType,
      tick: state.tick,
      secondsRemaining: state.secondsRemaining,
      selectedDecisionId: state.selectedDecision?.id ?? null,
      missionStatus: status,
      units: state.units.map((unit) => ({
        id: unit.id,
        side: unit.side,
        type: unit.type,
        label: unit.label,
        x: unit.x,
        y: unit.y,
        health: unit.health,
        range: unit.range,
      })),
    }),
  }).catch(() => {
    // Telemetry should never interrupt the live demo.
  });
}

function resetScenario() {
  if (state.units.length > 0 && !state.endLogged) {
    logSimulationEvent("reset", "reset_before_completion");
  }
  state.runId = createRunId();
  state.tick = 0;
  const terrain = terrainTemplates[Math.floor(Math.random() * terrainTemplates.length)];
  state.terrain = terrain;
  state.secondsRemaining = DEMO_SECONDS;
  state.selectedDecision = null;
  state.ended = false;
  state.dragging = null;
  state.lastDecisionRenderAt = 0;
  state.decisionSignature = "";
  state.renderedDecisions = [];
  state.fireTracers = [];
  state.popoverDrag = null;
  state.modelDecisions = [];
  state.modelOnline = false;
  state.modelFetchInFlight = false;
  state.lastModelFetchAt = 0;
  state.endLogged = false;
  hideRawDataPopover();
  reiterateButton.disabled = true;
  currentDecision.textContent = "Awaiting commander decision";

  state.units = [
    createUnit({
      id: "vip",
      side: "friendly",
      type: "VIP",
      label: "VIP",
      x: 145,
      y: randomBetween(450, 500),
      speed: 0.75,
      range: 70,
      health: 100,
    }),
    createUnit({
      id: "escort-1",
      side: "friendly",
      type: "MRAP",
      label: "SEC-1",
      x: 215,
      y: randomBetween(415, 475),
      speed: 0.95,
      range: 105,
      health: 100,
    }),
    createUnit({
      id: "escort-2",
      side: "friendly",
      type: "INF",
      label: "INF",
      x: 260,
      y: randomBetween(465, 520),
      speed: 0.82,
      range: 90,
      health: 100,
    }),
    createUnit({
      id: "scout-uas",
      side: "friendly",
      type: "UAS",
      label: "ISR",
      x: 340,
      y: randomBetween(330, 390),
      speed: 1.25,
      range: 160,
      health: 80,
    }),
    createUnit({
      id: "enemy-uas-1",
      side: "enemy",
      type: "UAS",
      label: "E-UAS",
      x: randomBetween(760, 880),
      y: randomBetween(100, 185),
      speed: 1.15,
      range: 125,
      health: 70,
    }),
    createUnit({
      id: "enemy-inf-1",
      side: "enemy",
      type: "INF",
      label: "CELL-A",
      x: randomBetween(715, 850),
      y: randomBetween(275, 370),
      speed: 0.7,
      range: 95,
      health: 90,
    }),
    createUnit({
      id: "enemy-inf-2",
      side: "enemy",
      type: "INF",
      label: "CELL-B",
      x: randomBetween(610, 750),
      y: randomBetween(150, 240),
      speed: 0.72,
      range: 95,
      health: 90,
    }),
  ];

  render();
  logSimulationEvent("started");
}

function getFriendlyUnits() {
  return state.units.filter((unit) => unit.side === "friendly" && unit.health > 0);
}

function getEnemyUnits() {
  return state.units.filter((unit) => unit.side === "enemy" && unit.health > 0);
}

function chooseEnemyTargets() {
  const vip = state.units.find((unit) => unit.id === "vip" && unit.health > 0);
  const friendly = getFriendlyUnits();
  const convoyCenter = vip || friendly[0];

  for (const enemy of getEnemyUnits()) {
    const target = enemy.type === "UAS" ? convoyCenter : nearestUnit(enemy, friendly);
    if (!target) continue;

    const flank = enemy.id.endsWith("2") ? -70 : 70;
    enemy.targetX = clamp(target.x + flank, 45, WIDTH - 45);
    enemy.targetY = clamp(target.y - 25, 45, HEIGHT - 45);
  }
}

function applySelectedDecision() {
  const friendly = getFriendlyUnits();
  const enemies = getEnemyUnits();
  const vip = friendly.find((unit) => unit.id === "vip");

  if (!state.selectedDecision || !vip) return;

  if (state.selectedDecision.id === "break-contact") {
    for (const unit of friendly) {
      unit.targetX = clamp(unit.x - 170, 50, WIDTH - 50);
      unit.targetY = clamp(unit.y + (unit.id === "scout-uas" ? -70 : 20), 50, HEIGHT - 50);
    }
  }

  if (state.selectedDecision.id === "counter-uas") {
    const drone = enemies.find((unit) => unit.type === "UAS") || enemies[0];
    for (const unit of friendly) {
      if (unit.id === "scout-uas" && drone) {
        unit.targetX = clamp(drone.x - 55, 50, WIDTH - 50);
        unit.targetY = clamp(drone.y + 45, 50, HEIGHT - 50);
      } else {
        unit.targetX = clamp(vip.x + randomBetween(-65, 65), 50, WIDTH - 50);
        unit.targetY = clamp(vip.y + randomBetween(-55, 55), 50, HEIGHT - 50);
      }
    }
  }

  if (state.selectedDecision.id === "screen-and-push") {
    for (const unit of friendly) {
      const offset = unit.id === "vip" ? 0 : randomBetween(-85, 85);
      unit.targetX = clamp(unit.x + 125, 50, WIDTH - 50);
      unit.targetY = clamp(unit.y + offset * 0.35, 50, HEIGHT - 50);
    }
  }
}

function moveUnitTowardTarget(unit) {
  unit.lastX = unit.x;
  unit.lastY = unit.y;
  const dx = unit.targetX - unit.x;
  const dy = unit.targetY - unit.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) return;

  const step = unit.speed * (unit.side === "enemy" ? 1.2 : 1);
  unit.x += (dx / length) * step;
  unit.y += (dy / length) * step;
  unit.x = clamp(unit.x, 30, WIDTH - 30);
  unit.y = clamp(unit.y, 30, HEIGHT - 30);
}

function resolveCombat() {
  const friendlies = getFriendlyUnits();
  const enemies = getEnemyUnits();

  for (const enemy of enemies) {
    const target = nearestUnit(enemy, friendlies);
    if (target && distance(enemy, target) < enemy.range) {
      target.health -= enemy.type === "UAS" ? 0.42 : 0.32;
      emitFireTracer(enemy, target);
    }
  }

  for (const friendly of friendlies) {
    const target = nearestUnit(friendly, enemies);
    if (target && distance(friendly, target) < friendly.range) {
      let effect = friendly.type === "UAS" ? 0.2 : 0.35;
      if (state.selectedDecision?.id === "counter-uas" && target.type === "UAS") {
        effect *= 2.8;
      }
      if (state.selectedDecision?.id === "screen-and-push") {
        effect *= 1.35;
      }
      target.health -= effect;
      emitFireTracer(friendly, target);
    }
  }

  for (const unit of state.units) {
    unit.health = clamp(unit.health, 0, 100);
  }
}

function computeThreat() {
  const friendly = getFriendlyUnits();
  const enemies = getEnemyUnits();
  if (!friendly.length || !enemies.length) return 0;
  const closest = Math.min(
    ...enemies.flatMap((enemy) => friendly.map((unit) => distance(enemy, unit)))
  );
  const enemyHealth = enemies.reduce((sum, unit) => sum + unit.health, 0);
  const friendlyHealth = friendly.reduce((sum, unit) => sum + unit.health, 0);
  const proximity = clamp((260 - closest) / 260, 0, 1);
  const ratio = clamp(enemyHealth / Math.max(1, friendlyHealth), 0, 1.7) / 1.7;
  return Math.round((proximity * 0.68 + ratio * 0.32) * 100);
}

function computeHeuristicDecisions() {
  const threat = computeThreat();
  const enemies = getEnemyUnits();
  const enemyDroneAlive = enemies.some((unit) => unit.type === "UAS");
  const vip = state.units.find((unit) => unit.id === "vip");
  const vipHealth = vip?.health ?? 0;

  const decisions = [
    {
      id: "counter-uas",
      title: "Prioritize counter-UAS intercept",
      score: clamp(76 + (enemyDroneAlive ? 12 : -18) + threat * 0.08, 1, 99),
      summary: "Move ISR and escort fires onto the hostile drone to reduce ambush coordination.",
      modelSource: "local heuristic fallback",
    },
    {
      id: "break-contact",
      title: "Break contact and reverse convoy",
      score: clamp(68 + threat * 0.2 + (vipHealth < 55 ? 14 : 0), 1, 99),
      summary: "Pull the VIP vehicle back through the cleared road segment while escorts cover.",
      modelSource: "local heuristic fallback",
    },
    {
      id: "screen-and-push",
      title: "Dismount screen and push through",
      score: clamp(72 - threat * 0.12 + (vipHealth > 70 ? 8 : -8), 1, 99),
      summary: "Use infantry and MRAP to screen the danger area while the VIP vehicle accelerates.",
      modelSource: "local heuristic fallback",
    },
  ];

  return decisions.sort((a, b) => b.score - a.score).slice(0, 3);
}

function computeDecisions() {
  return state.modelDecisions.length > 0 ? state.modelDecisions : computeHeuristicDecisions();
}

function buildModelRows(decision) {
  if (decision.rawRows) return decision.rawRows;

  const threat = computeThreat();
  const friendlies = getFriendlyUnits();
  const enemies = getEnemyUnits();
  const vip = state.units.find((unit) => unit.id === "vip");
  const enemyDroneAlive = enemies.some((unit) => unit.type === "UAS");
  const closestEnemy = friendlies.length && enemies.length
    ? Math.min(...enemies.flatMap((enemy) => friendlies.map((unit) => distance(enemy, unit))))
    : 0;

  return [
    ["decision_action", decision.title],
    ["success_probability", `${(decision.score / 100).toFixed(3)} (${Math.round(decision.score)}%)`],
    ["model_source", decision.modelSource ?? "local heuristic fallback"],
    ["target_column", "attacker_success"],
    ["tactical_posture", decision.id],
    ["war4_theater", "Modern local demo"],
    ["terrain_primary", "R"],
    ["weather_primary", "D"],
    ["front", threat > 45 ? 1 : 0],
    ["depth", threat > 65 ? 1 : 0],
    ["aeroa", enemyDroneAlive ? -1 : 1],
    ["surpa", threat > 55 ? -1 : 0],
    ["cea", Math.round((friendlies.length - enemies.length) * 0.5)],
    ["leada", state.selectedDecision?.id === decision.id ? 1 : 0],
    ["intela", enemyDroneAlive ? 0 : 1],
    ["techa", decision.id === "counter-uas" ? 1 : 0],
    ["inita", decision.id === "screen-and-push" ? 1 : 0],
    ["quala", Math.round((vip?.health ?? 0) / 25) / 2],
    ["resa", decision.id === "break-contact" ? 1 : 0],
    ["mobila", decision.id === "screen-and-push" ? 1 : 0],
    ["aira", enemyDroneAlive ? -1 : 0],
    ["wxa", 0],
    ["terra", 1],
    ["duration2", ((DEMO_SECONDS - state.secondsRemaining) / 86400).toFixed(5)],
    ["closest_enemy_m", Math.round(closestEnemy * 2.5)],
    ["friendly_combat_power", Math.round(friendlies.reduce((sum, unit) => sum + unit.health, 0))],
    ["enemy_combat_power", Math.round(enemies.reduce((sum, unit) => sum + unit.health, 0))],
  ];
}

function showRawDataPopover(decision) {
  rawDataTitle.textContent = decision.title;
  rawDataBody.replaceChildren();

  const intro = document.createElement("p");
  intro.className = "popover-note";
  intro.textContent = decision.rawRows
    ? "Raw Logit Hierarchical Regression output for this option, generated from live simulation sensor features."
    : "Backend is offline, so this is the local heuristic fallback view.";

  const table = document.createElement("table");
  table.className = "raw-data-table";
  table.innerHTML = "<thead><tr><th>Field</th><th>Value</th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (const [field, value] of buildModelRows(decision)) {
    const row = document.createElement("tr");
    const key = document.createElement("th");
    key.scope = "row";
    key.textContent = field;
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.append(key, cell);
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  rawDataBody.append(intro, table);
  rawDataPopover.hidden = false;
}

function hideRawDataPopover() {
  if (rawDataPopover) rawDataPopover.hidden = true;
}

function updateSituationText() {
  const threat = computeThreat();
  const friendlyCount = getFriendlyUnits().length;
  const enemyCount = getEnemyUnits().length;

  if (state.ended) {
    if (friendlyCount === 0) {
      situationTitle.textContent = "Mission failed: friendly force combat ineffective";
      battlefieldStatus.textContent = "Enemy force controls the road segment";
    } else if (enemyCount === 0) {
      situationTitle.textContent = "Mission success: hostile ambush neutralized";
      battlefieldStatus.textContent = "Friendly force controls the road segment";
    } else {
      situationTitle.textContent = "Demo clock expired: final decision frozen";
      battlefieldStatus.textContent = "Scenario complete";
    }
    situationCopy.textContent = `Final decision: ${state.selectedDecision?.title ?? "No decision selected"}.`;
    return;
  }

  if (threat > 72) {
    situationTitle.textContent = "Hostile ambush is inside effective range";
    battlefieldStatus.textContent = "High threat";
  } else if (threat > 42) {
    situationTitle.textContent = "Enemy elements converging on VIP convoy";
    battlefieldStatus.textContent = "Ambush developing";
  } else {
    situationTitle.textContent = "VIP convoy entering hostile road segment";
    battlefieldStatus.textContent = "Contact likely";
  }

  situationCopy.textContent =
    "Friendly convoy is moving through restricted terrain while hostile drone and infantry cells coordinate an ambush.";
}

function serializeSimulationState() {
  return {
    runId: state.runId,
    tick: state.tick,
    secondsRemaining: state.secondsRemaining,
    selectedDecisionId: state.selectedDecision?.id ?? null,
    units: state.units.map((unit) => ({
      id: unit.id,
      side: unit.side,
      type: unit.type,
      label: unit.label,
      x: unit.x,
      y: unit.y,
      health: unit.health,
      range: unit.range,
    })),
  };
}

async function refreshModelDecisions(force = false) {
  const now = Date.now();
  if (state.modelFetchInFlight || (!force && now - state.lastModelFetchAt < 1100)) {
    return;
  }

  state.modelFetchInFlight = true;
  state.lastModelFetchAt = now;
  try {
    const response = await fetch(`${API_BASE_URL}/api/decisions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(serializeSimulationState()),
    });
    if (!response.ok) throw new Error(`Model backend returned ${response.status}`);
    const payload = await response.json();
    state.modelDecisions = payload.decisions.map((decision) => ({
      id: decision.id,
      title: decision.title,
      summary: decision.summary,
      score: decision.score,
      successProbability: decision.success_probability,
      modelSource: decision.model_source,
      mercuryUsed: decision.mercury_used,
      mercurySummary: decision.mercury_summary,
      rawRows: decision.raw_rows,
      features: decision.features,
    }));
    state.modelOnline = true;
    state.lastDecisionRenderAt = 0;
    render();
  } catch {
    state.modelOnline = false;
    state.modelDecisions = [];
  } finally {
    state.modelFetchInFlight = false;
  }
}

function tick() {
  if (state.ended) return;

  state.tick += 1;
  refreshModelDecisions();
  state.secondsRemaining -= TICK_MS / 1000;
  chooseEnemyTargets();
  applySelectedDecision();

  for (const unit of state.units.filter((item) => item.health > 0)) {
    if (state.dragging?.id !== unit.id) moveUnitTowardTarget(unit);
  }

  resolveCombat();

  if (getFriendlyUnits().length === 0 || getEnemyUnits().length === 0 || state.secondsRemaining <= 0) {
    state.ended = true;
    reiterateButton.disabled = false;
    if (!state.endLogged) {
      state.endLogged = true;
      logSimulationEvent("ended", missionStatus());
    }
  }

  render();
}

function svgPointFromEvent(event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function setAttributes(element, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value);
  }
}

function el(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  setAttributes(node, attrs);
  return node;
}

function renderTerrain() {
  terrainLayer.replaceChildren();
  terrainLayer.appendChild(el("path", { d: state.terrain.road, class: "road" }));
  terrainLayer.appendChild(el("path", { d: state.terrain.road, class: "road-line" }));

  for (const [x, y, width, height] of state.terrain.cover) {
    terrainLayer.appendChild(el("rect", {
      x,
      y,
      width,
      height,
      rx: 18,
      class: "cover",
    }));
  }

  const [x, y, width, height] = state.terrain.danger;
  terrainLayer.appendChild(el("rect", {
    x,
    y,
    width,
    height,
    rx: 28,
    class: "danger-zone",
  }));
}

function renderSensors() {
  sensorLayer.replaceChildren();
  for (const unit of state.units.filter((item) => item.health > 0 && item.side === "friendly")) {
    if (unit.type === "UAS" || unit.id === "escort-1") {
      sensorLayer.appendChild(el("circle", {
        cx: unit.x,
        cy: unit.y,
        r: unit.type === "UAS" ? 155 : 105,
        class: "sensor-ring",
      }));
    }
  }
}

function renderMovement() {
  movementLayer.replaceChildren();
  for (const unit of state.units.filter((item) => item.health > 0)) {
    const moved = Math.hypot(unit.targetX - unit.x, unit.targetY - unit.y) > 12;
    if (!moved) continue;
    movementLayer.appendChild(el("line", {
      x1: unit.x,
      y1: unit.y,
      x2: unit.targetX,
      y2: unit.targetY,
      class: `movement-line ${unit.side}`,
    }));
  }
}

function renderFireTracers() {
  const now = Date.now();
  state.fireTracers = state.fireTracers.filter((tracer) => tracer.expiresAt > now);
  fireLayer.replaceChildren();

  for (const tracer of state.fireTracers) {
    const remaining = clamp((tracer.expiresAt - now) / 340, 0, 1);
    const blinkOn = Math.floor((340 - (tracer.expiresAt - now)) / 70) % 2 === 0;
    const line = el("line", {
      x1: tracer.x1,
      y1: tracer.y1,
      x2: tracer.x2,
      y2: tracer.y2,
      class: "fire-tracer",
      opacity: blinkOn ? remaining : 0.08,
      "stroke-width": blinkOn ? 5 : 2,
    });
    fireLayer.appendChild(line);
  }
}

function unitFrame(unit) {
  if (unit.side === "enemy") {
    return el("polygon", {
      points: "0,-23 32,0 0,23 -32,0",
      class: "unit-frame",
    });
  }
  if (unit.type === "UAS") {
    return el("circle", {
      cx: 0,
      cy: 0,
      r: 24,
      class: "unit-frame",
    });
  }
  return el("rect", {
    x: -33,
    y: -22,
    width: 66,
    height: 44,
    rx: 4,
    class: "unit-frame",
  });
}

function renderUnits() {
  unitLayer.replaceChildren();
  for (const unit of state.units) {
    const group = el("g", {
      class: `unit ${unit.side} ${unit.health <= 0 ? "destroyed" : ""} ${state.dragging?.id === unit.id ? "dragging" : ""}`,
      transform: `translate(${unit.x} ${unit.y})`,
      "data-id": unit.id,
      tabindex: "0",
      role: "button",
      "aria-label": `${unit.side} ${unit.label} ${Math.round(unit.health)} percent health`,
    });

    group.appendChild(unitFrame(unit));
    group.appendChild(el("line", {
      x1: -16,
      y1: unit.side === "enemy" ? 0 : -22,
      x2: 16,
      y2: unit.side === "enemy" ? 0 : -22,
      stroke: unit.side === "enemy" ? "#ff867c" : "#64b5f6",
      "stroke-width": 3,
    }));

    const type = el("text", { x: 0, y: 4 });
    type.textContent = unit.type;
    group.appendChild(type);

    const label = el("text", { x: 0, y: 37, class: "label" });
    label.textContent = unit.label;
    group.appendChild(label);

    group.appendChild(el("rect", {
      x: -29,
      y: 27,
      width: 58,
      height: 6,
      rx: 3,
      class: "health-bar-bg",
    }));
    group.appendChild(el("rect", {
      x: -29,
      y: 27,
      width: 58 * (unit.health / 100),
      height: 6,
      rx: 3,
      class: `health-bar ${unit.health < 40 ? "low" : ""}`,
    }));

    group.addEventListener("pointerdown", (event) => {
      if (unit.health <= 0 || state.ended) return;
      const point = svgPointFromEvent(event);
      state.dragging = {
        id: unit.id,
        dx: point.x - unit.x,
        dy: point.y - unit.y,
      };
      group.setPointerCapture(event.pointerId);
      render();
    });

    unitLayer.appendChild(group);
  }
}

function renderDecisionList() {
  const decisions = computeDecisions();
  state.renderedDecisions = decisions;
  const signature = decisions
    .map((decision) => `${decision.id}:${Math.round(decision.score)}`)
    .join("|");
  const now = Date.now();
  if (
    decisionList.childElementCount > 0
    && signature === state.decisionSignature
    && now - state.lastDecisionRenderAt < 1500
  ) {
    return;
  }

  state.decisionSignature = signature;
  state.lastDecisionRenderAt = now;

  if (decisionList.childElementCount === 0) {
    for (let index = 0; index < 3; index += 1) {
      const card = document.createElement("article");
      card.className = "decision-card";

      const heading = document.createElement("h3");
      const title = document.createElement("span");
      title.className = "decision-title";
      const score = document.createElement("span");
      score.className = "score";
      heading.append(title, score);

      const summary = document.createElement("p");
      summary.className = "decision-summary";

      const actions = document.createElement("div");
      actions.className = "decision-actions";

      const inspectButton = document.createElement("button");
      inspectButton.type = "button";
      inspectButton.className = "secondary compact";
      inspectButton.textContent = "Raw data";
      inspectButton.addEventListener("click", () => {
        const decision = state.renderedDecisions[index];
        if (!decision) return;
        showRawDataPopover(decision);
      });

      const selectButton = document.createElement("button");
      selectButton.type = "button";
      selectButton.className = "select-decision";
      selectButton.textContent = "Select option";
      selectButton.addEventListener("click", () => {
        const decision = state.renderedDecisions[index];
        if (!decision) return;
        state.selectedDecision = decision;
        state.lastDecisionRenderAt = 0;
        currentDecision.textContent = `${decision.title} (${Math.round(decision.score)}% estimated success)`;
        logSimulationEvent("selected_decision");
        applySelectedDecision();
        render();
      });

      actions.append(inspectButton, selectButton);
      card.append(heading, summary, actions);
      decisionList.appendChild(card);
    }
  }

  for (const [index, decision] of decisions.entries()) {
    const card = decisionList.children[index];
    if (!card) continue;
    const isSelected = state.selectedDecision?.id === decision.id;
    card.classList.toggle("selected", isSelected);
    card.querySelector(".decision-title").textContent = decision.title;
    card.querySelector(".score").textContent = `${Math.round(decision.score)}%`;
    card.querySelector(".decision-summary").textContent = decision.summary;
    card.querySelector(".select-decision").textContent = isSelected ? "Selected" : "Select option";
  }
}

function renderSensorSummary() {
  const threat = computeThreat();
  const friendlyHealth = getFriendlyUnits().reduce((sum, unit) => sum + unit.health, 0);
  const enemyHealth = getEnemyUnits().reduce((sum, unit) => sum + unit.health, 0);
  const enemyDrone = getEnemyUnits().some((unit) => unit.type === "UAS") ? "Tracked" : "Neutralized";

  sensorSummary.replaceChildren();
  const rows = [
    ["Threat index", `${threat}/100`],
    [
      "Decision source",
      state.modelOnline
        ? `HF Logit model${state.modelDecisions.some((decision) => decision.mercuryUsed) ? " + Mercury II" : ""}`
        : "Heuristic fallback",
    ],
    ["Friendly combat power", Math.round(friendlyHealth).toString()],
    ["Enemy combat power", Math.round(enemyHealth).toString()],
    ["Hostile drone", enemyDrone],
  ];

  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    sensorSummary.append(dt, dd);
  }
}

function renderClock() {
  const seconds = Math.max(0, Math.ceil(state.secondsRemaining));
  const min = Math.floor(seconds / 60).toString().padStart(2, "0");
  const sec = (seconds % 60).toString().padStart(2, "0");
  demoClock.textContent = `${min}:${sec}`;
}

function render() {
  renderTerrain();
  renderSensors();
  renderMovement();
  renderFireTracers();
  renderUnits();
  renderDecisionList();
  renderSensorSummary();
  renderClock();
  updateSituationText();
}

svg.addEventListener("pointermove", (event) => {
  if (!state.dragging || state.ended) return;
  const unit = state.units.find((item) => item.id === state.dragging.id);
  if (!unit) return;
  const point = svgPointFromEvent(event);
  unit.x = clamp(point.x - state.dragging.dx, 30, WIDTH - 30);
  unit.y = clamp(point.y - state.dragging.dy, 30, HEIGHT - 30);
  unit.targetX = unit.x;
  unit.targetY = unit.y;
  render();
});

svg.addEventListener("pointerup", () => {
  state.dragging = null;
  render();
});

svg.addEventListener("pointercancel", () => {
  state.dragging = null;
  render();
});

rawDataClose.addEventListener("click", hideRawDataPopover);

rawDataDragHandle.addEventListener("pointerdown", (event) => {
  if (event.target === rawDataClose) return;
  const rect = rawDataPopover.getBoundingClientRect();
  state.popoverDrag = {
    dx: event.clientX - rect.left,
    dy: event.clientY - rect.top,
  };
  rawDataDragHandle.setPointerCapture(event.pointerId);
});

window.addEventListener("pointermove", (event) => {
  if (!state.popoverDrag) return;
  const left = clamp(event.clientX - state.popoverDrag.dx, 8, window.innerWidth - 260);
  const top = clamp(event.clientY - state.popoverDrag.dy, 8, window.innerHeight - 120);
  rawDataPopover.style.left = `${left}px`;
  rawDataPopover.style.top = `${top}px`;
  rawDataPopover.style.right = "auto";
});

window.addEventListener("pointerup", () => {
  state.popoverDrag = null;
});

resetButton.addEventListener("click", () => {
  logSimulationEvent("reset", missionStatus());
  state.endLogged = true;
  resetScenario();
});
reiterateButton.addEventListener("click", () => {
  logSimulationEvent("reiterate", missionStatus());
  state.endLogged = true;
  resetScenario();
});

resetScenario();
refreshModelDecisions(true);
state.tickHandle = window.setInterval(tick, TICK_MS);

