const WIDTH = 1000;
const HEIGHT = 620;
const TICK_MS = 240;
const DEMO_SECONDS = 5 * 60;
const API_BASE_URL = "http://127.0.0.1:8000";
const VIP_EXTRACTION_X = 900;
const LOCAL_SENSOR_RANGE = 220;

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
const selectedDecisionTrend = document.querySelector("#selected-decision-trend");
const battlefieldStatus = document.querySelector("#battlefield-status");
const demoClock = document.querySelector("#demo-clock");
const resetButton = document.querySelector("#reset-button");
const reiterateButton = document.querySelector("#reiterate-button");
const rawDataPopover = document.querySelector("#raw-data-popover");
const rawDataBody = document.querySelector("#raw-data-body");
const rawDataTitle = document.querySelector("#raw-data-title");
const rawDataClose = document.querySelector("#raw-data-close");
const rawDataDragHandle = document.querySelector("#raw-data-drag-handle");
const friendlyCountInput = document.querySelector("#friendly-count");
const enemyCountInput = document.querySelector("#enemy-count");
const droneAmbushInput = document.querySelector("#drone-ambush");

const UNIT_STATS = {
  VIP: { health: 100, speed: 0.75, range: 70, damage: 0.18 },
  MRAP: { health: 100, speed: 0.95, range: 105, damage: 0.35 },
  INF: { health: 90, speed: 0.82, range: 95, damage: 0.32 },
  UAS: { health: 20, speed: 2.35, range: 28, damage: 140 },
};

const UNIT_COUNT_DEFAULTS = {
  friendly: 4,
  enemy: 3,
};

const GROUND_UNIT_RADIUS = 26;

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

const BUILDING_WALLS = [
  { x: 0, y: 365, width: 130, height: 150 },
  { x: 130, y: 386, width: 82, height: 115 },
  { x: 205, y: 500, width: 88, height: 90 },
  { x: 300, y: 472, width: 105, height: 105 },
  { x: 406, y: 228, width: 92, height: 76 },
  { x: 510, y: 202, width: 116, height: 94 },
  { x: 712, y: 224, width: 124, height: 104 },
  { x: 808, y: 286, width: 74, height: 112 },
  { x: 888, y: 204, width: 96, height: 148 },
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
  selectedTrend: null,
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

function distanceToMovementSegment(unit, target) {
  const start = { x: unit.lastX ?? unit.x, y: unit.lastY ?? unit.y };
  const end = { x: unit.x, y: unit.y };
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return distance(unit, target);

  const t = clamp(
    ((target.x - start.x) * dx + (target.y - start.y) * dy) / lengthSquared,
    0,
    1
  );
  return Math.hypot(start.x + dx * t - target.x, start.y + dy * t - target.y);
}

function withinFpvImpactRange(drone, target) {
  return distance(drone, target) < drone.range || distanceToMovementSegment(drone, target) < drone.range;
}

function isGroundUnit(unit) {
  return unit.type !== "UAS";
}

function paddedRect(rect, padding) {
  return {
    x: rect.x - padding,
    y: rect.y - padding,
    width: rect.width + padding * 2,
    height: rect.height + padding * 2,
  };
}

function pointInsideRect(point, rect) {
  return (
    point.x >= rect.x
    && point.x <= rect.x + rect.width
    && point.y >= rect.y
    && point.y <= rect.y + rect.height
  );
}

function isGroundPositionBlocked(position) {
  return BUILDING_WALLS.some((wall) => (
    pointInsideRect(position, paddedRect(wall, GROUND_UNIT_RADIUS))
  ));
}

function segmentIntersectsRect(a, b, rect) {
  const samples = 28;
  for (let index = 1; index < samples; index += 1) {
    const t = index / samples;
    const point = {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
    };
    if (pointInsideRect(point, rect)) return true;
  }
  return false;
}

function hasBuildingLineOfSightBlock(source, target) {
  return BUILDING_WALLS.some((wall) => segmentIntersectsRect(source, target, wall));
}

function pushPointOutsideRect(point, rect) {
  const left = Math.abs(point.x - rect.x);
  const right = Math.abs(rect.x + rect.width - point.x);
  const top = Math.abs(point.y - rect.y);
  const bottom = Math.abs(rect.y + rect.height - point.y);
  const nearest = Math.min(left, right, top, bottom);

  if (nearest === left) return { x: rect.x - 1, y: point.y };
  if (nearest === right) return { x: rect.x + rect.width + 1, y: point.y };
  if (nearest === top) return { x: point.x, y: rect.y - 1 };
  return { x: point.x, y: rect.y + rect.height + 1 };
}

function constrainGroundPosition(unit, position) {
  if (!isGroundUnit(unit)) return position;

  let next = {
    x: clamp(position.x, 30, WIDTH - 30),
    y: clamp(position.y, 30, HEIGHT - 30),
  };

  for (const wall of BUILDING_WALLS) {
    const rect = paddedRect(wall, GROUND_UNIT_RADIUS);
    if (pointInsideRect(next, rect)) {
      next = pushPointOutsideRect(next, rect);
    }
  }

  return {
    x: clamp(next.x, 30, WIDTH - 30),
    y: clamp(next.y, 30, HEIGHT - 30),
  };
}

function nextGroundStep(unit, dx, dy, length, step) {
  if (!isGroundUnit(unit)) {
    return {
      x: clamp(unit.x + (dx / length) * step, 30, WIDTH - 30),
      y: clamp(unit.y + (dy / length) * step, 30, HEIGHT - 30),
    };
  }

  const ux = dx / length;
  const uy = dy / length;
  const direct = {
    x: clamp(unit.x + ux * step, 30, WIDTH - 30),
    y: clamp(unit.y + uy * step, 30, HEIGHT - 30),
  };
  if (!isGroundPositionBlocked(direct)) return direct;

  const sideStep = step * 1.25;
  const forwardStep = step * 0.35;
  const candidates = [
    { x: unit.x - uy * sideStep + ux * forwardStep, y: unit.y + ux * sideStep + uy * forwardStep },
    { x: unit.x + uy * sideStep + ux * forwardStep, y: unit.y - ux * sideStep + uy * forwardStep },
    { x: unit.x - uy * sideStep, y: unit.y + ux * sideStep },
    { x: unit.x + uy * sideStep, y: unit.y - ux * sideStep },
  ].map((point) => ({
    x: clamp(point.x, 30, WIDTH - 30),
    y: clamp(point.y, 30, HEIGHT - 30),
  }));

  const viable = candidates.filter((point) => !isGroundPositionBlocked(point));
  if (viable.length > 0) {
    return viable.sort((a, b) => (
      Math.hypot(a.x - unit.targetX, a.y - unit.targetY)
      - Math.hypot(b.x - unit.targetX, b.y - unit.targetY)
    ))[0];
  }

  return constrainGroundPosition(unit, direct);
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

function unitHealthPercent(unit) {
  return Math.round((unit.health / unit.maxHealth) * 100);
}

function configuredUnitCount(side) {
  const input = side === "friendly" ? friendlyCountInput : enemyCountInput;
  const fallback = UNIT_COUNT_DEFAULTS[side];
  return clamp(Number.parseInt(input?.value ?? fallback, 10) || fallback, 1, 8);
}

function droneAmbushEnabled() {
  return Boolean(droneAmbushInput?.checked);
}

function createUnit({ id, side, type, label, x, y, speed, range, health }) {
  const stats = UNIT_STATS[type] ?? UNIT_STATS.INF;
  const maxHealth = health ?? stats.health;
  return {
    id,
    side,
    type,
    label,
    x,
    y,
    speed: speed ?? stats.speed,
    range: range ?? stats.range,
    health: maxHealth,
    maxHealth,
    damage: stats.damage,
    targetX: x,
    targetY: y,
    lastX: x,
    lastY: y,
  };
}

function missionStatus() {
  const friendlyAlive = getFriendlyUnits().length;
  const enemyAlive = getEnemyUnits().length;
  const vipAlive = state.units.some((unit) => unit.id === "vip" && unit.health > 0);
  if (!vipAlive) return "enemy_victory";
  if (vipExtracted()) return "friendly_extraction";
  if (friendlyAlive === 0) return "enemy_victory";
  if (enemyAlive === 0) return "friendly_victory";
  if (state.secondsRemaining <= 0) return "clock_expired";
  return "running";
}

function vipExtracted() {
  const vip = state.units.find((unit) => unit.id === "vip" && unit.health > 0);
  return Boolean(vip && vip.x >= VIP_EXTRACTION_X);
}

function endStateDetails() {
  const vipKilled = !state.units.some((unit) => unit.id === "vip" && unit.health > 0);
  const extracted = vipExtracted();
  const friendlyTotal = state.units.filter((unit) => unit.side === "friendly").length;
  const enemyTotal = state.units.filter((unit) => unit.side === "enemy").length;
  const friendlyAlive = getFriendlyUnits().length;
  const enemyAlive = getEnemyUnits().length;
  return {
    vipKilled,
    vipExtracted: extracted,
    allFriendliesEliminated: friendlyAlive === 0,
    allEnemiesEliminated: enemyAlive === 0,
    friendlyAlive,
    friendlyTotal,
    enemyAlive,
    enemyTotal,
  };
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

function unitTypeForSideIndex(side, index) {
  if (side === "friendly") {
    if (index === 0) return "VIP";
    if (index === 1) return "MRAP";
    if (index === 2) return "INF";
    if (index === 3) return "UAS";
    return index % 3 === 1 ? "MRAP" : "INF";
  }

  if (index === 0) return "UAS";
  if (index % 3 === 0) return "MRAP";
  return "INF";
}

function labelForUnit(side, type, index) {
  if (side === "friendly" && type === "VIP") return "VIP";
  if (side === "friendly" && type === "UAS") return "F-FPV";
  if (side === "friendly" && type === "MRAP") return `SEC-${index}`;
  if (side === "friendly") return `INF-${index}`;
  if (type === "UAS") return `E-FPV-${index + 1}`;
  if (type === "MRAP") return `E-VEH-${index + 1}`;
  return `CELL-${String.fromCharCode(65 + (index % 26))}`;
}

function displayUnitType(unit) {
  return unit.type === "UAS" ? "FPV" : unit.type;
}

function spawnFriendlyUnits(count) {
  const units = [];
  for (let index = 0; index < count; index += 1) {
    const type = unitTypeForSideIndex("friendly", index);
    units.push(createUnit({
      id: index === 0 ? "vip" : `friendly-${type.toLowerCase()}-${index}`,
      side: "friendly",
      type,
      label: labelForUnit("friendly", type, index),
      x: clamp(115 + index * 56 + randomBetween(-10, 10), 55, 380),
      y: clamp(342 + index * 10 + randomBetween(-8, 8), 60, HEIGHT - 60),
    }));
  }
  return units;
}

function spawnEnemyUnits(count) {
  const units = [];
  const useDroneAmbush = droneAmbushEnabled();
  for (let index = 0; index < count; index += 1) {
    const isController = useDroneAmbush && index === count - 1;
    const type = useDroneAmbush ? (isController ? "INF" : "UAS") : unitTypeForSideIndex("enemy", index);
    const upperLane = index % 2 === 0;
    units.push(createUnit({
      id: isController ? "enemy-drone-controller" : `enemy-${type.toLowerCase()}-${index + 1}`,
      side: "enemy",
      type,
      label: isController ? "CTRL" : labelForUnit("enemy", type, index),
      x: clamp(randomBetween(610, 875) - index * 18, 470, WIDTH - 60),
      y: upperLane ? randomBetween(110, 230) : randomBetween(275, 390),
    }));
  }
  return units;
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
  state.selectedTrend = null;
  hideRawDataPopover();
  reiterateButton.disabled = true;
  currentDecision.textContent = "Awaiting commander decision";
  selectedDecisionTrend.textContent = "Selected trend: awaiting selection";

  state.units = [
    ...spawnFriendlyUnits(configuredUnitCount("friendly")),
    ...spawnEnemyUnits(configuredUnitCount("enemy")),
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
  const friendly = getFriendlyUnits();

  for (const enemy of getEnemyUnits()) {
    const target = nearestUnit(enemy, friendly);
    if (!target) continue;

    if (enemy.type === "UAS") {
      enemy.targetX = clamp(target.x, 45, WIDTH - 45);
      enemy.targetY = clamp(target.y, 45, HEIGHT - 45);
      continue;
    }

    const flank = enemy.id.endsWith("2") ? -70 : 70;
    enemy.targetX = clamp(target.x + flank, 45, WIDTH - 45);
    enemy.targetY = clamp(target.y - 25, 45, HEIGHT - 45);
  }
}

function chooseFriendlyDroneTargets() {
  const enemies = getEnemyUnits();
  if (!enemies.length) return;

  for (const drone of getFriendlyUnits().filter((unit) => unit.type === "UAS")) {
    const target = state.selectedDecision?.id === "counter-uas"
      ? enemies.find((unit) => unit.type === "UAS") ?? nearestUnit(drone, enemies)
      : nearestUnit(drone, enemies);
    if (!target) continue;
    drone.targetX = target.x;
    drone.targetY = target.y;
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
      unit.targetY = clamp(unit.y + (unit.type === "UAS" ? -70 : 20), 50, HEIGHT - 50);
    }
  }

  if (state.selectedDecision.id === "counter-uas") {
    const drone = enemies.find((unit) => unit.type === "UAS") || enemies[0];
    for (const unit of friendly) {
      if (unit.type === "UAS" && drone) {
        unit.targetX = clamp(drone.x, 50, WIDTH - 50);
        unit.targetY = clamp(drone.y, 50, HEIGHT - 50);
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

  if (state.selectedDecision.id === "shift-vip-to-cover") {
    const coverX = WIDTH * 0.42;
    const coverY = HEIGHT * 0.72;
    for (const unit of friendly) {
      if (unit.id === "vip") {
        unit.targetX = clamp(coverX, 50, WIDTH - 50);
        unit.targetY = clamp(coverY, 50, HEIGHT - 50);
      } else {
        unit.targetX = clamp(coverX + randomBetween(-90, 90), 50, WIDTH - 50);
        unit.targetY = clamp(coverY + randomBetween(-70, 70), 50, HEIGHT - 50);
      }
    }
  }

  if (state.selectedDecision.id === "hold-defensive-perimeter") {
    for (const unit of friendly) {
      unit.targetX = clamp(vip.x + randomBetween(-95, 95), 50, WIDTH - 50);
      unit.targetY = clamp(vip.y + randomBetween(-75, 75), 50, HEIGHT - 50);
    }
  }

  if (state.selectedDecision.id === "call-for-reinforcement") {
    for (const unit of friendly) {
      unit.targetX = clamp(unit.x + randomBetween(-35, 35), 50, WIDTH - 50);
      unit.targetY = clamp(unit.y + randomBetween(-35, 35), 50, HEIGHT - 50);
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

  const step = unit.speed;
  const next = nextGroundStep(unit, dx, dy, length, step);
  unit.x = next.x;
  unit.y = next.y;
}

function resolveCombat() {
  const friendlies = getFriendlyUnits();
  const enemies = getEnemyUnits();

  for (const enemy of enemies) {
    const target = nearestUnit(enemy, friendlies);
    if (enemy.type === "UAS") {
      if (!target || !withinFpvImpactRange(enemy, target)) continue;
      target.health -= enemy.damage;
      enemy.health = 0;
      emitFireTracer(enemy, target);
    } else if (target && distance(enemy, target) < enemy.range && !hasBuildingLineOfSightBlock(enemy, target)) {
      target.health -= enemy.damage;
      emitFireTracer(enemy, target);
    }
  }

  for (const friendly of friendlies) {
    const target = nearestUnit(friendly, enemies);
    let effect = friendly.damage;
    if (friendly.type === "UAS") {
      if (!target || !withinFpvImpactRange(friendly, target)) continue;
      target.health -= effect;
      friendly.health = 0;
      emitFireTracer(friendly, target);
    } else if (target && distance(friendly, target) < friendly.range && !hasBuildingLineOfSightBlock(friendly, target)) {
      if (state.selectedDecision?.id === "screen-and-push") effect *= 1.35;
      target.health -= effect;
      emitFireTracer(friendly, target);
    }
  }

  for (const unit of state.units) {
    unit.health = clamp(unit.health, 0, unit.maxHealth);
  }

  eliminateEnemyDronesIfControllerKilled();
}

function eliminateEnemyDronesIfControllerKilled() {
  if (!droneAmbushEnabled()) return;
  const controllerAlive = state.units.some((unit) => (
    unit.id === "enemy-drone-controller" && unit.health > 0
  ));
  if (controllerAlive) return;

  for (const unit of state.units) {
    if (unit.side === "enemy" && unit.type === "UAS") {
      unit.health = 0;
    }
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
      title: "Prioritize FPV drone intercept",
      score: clamp(76 + (enemyDroneAlive ? 12 : -18) + threat * 0.08, 1, 99),
      summary: "Commit friendly FPV drone and escort fires against the hostile drone before it reaches the VIP.",
      modelSource: "local heuristic fallback",
    },
    {
      id: "break-contact",
      title: "Break contact and extract VIP",
      score: clamp(68 + threat * 0.2 + (vipHealth < 55 ? 14 : 0), 1, 99),
      summary: "Pull the VIP vehicle back through the cleared road segment while escorts cover the withdrawal.",
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

function trendLabelFromSlope(slope) {
  if (Math.abs(slope) < 0.25) return "steady";
  const sign = slope > 0 ? "+" : "";
  return `${sign}${slope.toFixed(1)} pp/min`;
}

function updateSelectedDecisionTrend(decisions) {
  if (!state.selectedDecision) {
    selectedDecisionTrend.textContent = "Selected trend: awaiting selection";
    selectedDecisionTrend.classList.remove("up", "down");
    return;
  }

  const current = decisions.find((decision) => decision.id === state.selectedDecision.id)
    ?? state.selectedDecision;
  const now = Date.now();
  const score = Number(current.score) || 0;
  const previous = state.selectedTrend;
  let slope = previous?.slope ?? 0;

  if (!previous || previous.id !== current.id) {
    slope = 0;
    state.selectedTrend = { id: current.id, score, at: now, slope };
  } else if (Math.abs(score - previous.score) >= 0.05) {
    const elapsedMinutes = Math.max((now - previous.at) / 60000, 1 / 60);
    slope = clamp((score - previous.score) / elapsedMinutes, -99, 99);
    state.selectedTrend = { id: current.id, score, at: now, slope };
  }

  currentDecision.textContent = `${current.title} (${Math.round(score)}% estimated success)`;
  selectedDecisionTrend.textContent = `Selected trend: ${trendLabelFromSlope(slope)}`;
  selectedDecisionTrend.classList.toggle("up", slope > 0.25);
  selectedDecisionTrend.classList.toggle("down", slope < -0.25);
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
    const details = endStateDetails();
    if (details.vipKilled) {
      situationTitle.textContent = "Mission failed: VIP lost";
      battlefieldStatus.textContent = "VIP casualty";
    } else if (friendlyCount === 0) {
      situationTitle.textContent = "Mission failed: friendly force combat ineffective";
      battlefieldStatus.textContent = "Enemy force controls the road segment";
    } else if (details.vipExtracted) {
      situationTitle.textContent = "Mission success: VIP cleared the ambush";
      battlefieldStatus.textContent = "VIP extracted";
    } else if (enemyCount === 0) {
      situationTitle.textContent = "Mission success: hostile ambush neutralized";
      battlefieldStatus.textContent = "Friendly force controls the road segment";
    } else {
      situationTitle.textContent = "Demo clock expired: final decision frozen";
      battlefieldStatus.textContent = "Scenario complete";
    }
    situationCopy.textContent = [
      `Final decision: ${state.selectedDecision?.title ?? "No decision selected"}.`,
      `VIP killed: ${details.vipKilled ? "yes" : "no"}.`,
      `VIP extracted: ${details.vipExtracted ? "yes" : "no"}.`,
      `All friendlies eliminated: ${details.allFriendliesEliminated ? "yes" : "no"} (${details.friendlyAlive}/${details.friendlyTotal} alive).`,
      `All enemies eliminated: ${details.allEnemiesEliminated ? "yes" : "no"} (${details.enemyAlive}/${details.enemyTotal} alive).`,
    ].join(" ");
    return;
  }

  if (threat > 72) {
    situationTitle.textContent = "Hostile ambush is inside effective range";
    battlefieldStatus.textContent = "High threat";
  } else if (threat > 42) {
    situationTitle.textContent = "Enemy elements converging on VIP";
    battlefieldStatus.textContent = "Ambush developing";
  } else {
    situationTitle.textContent = "VIP entering hostile road segment";
    battlefieldStatus.textContent = "Contact likely";
  }

  situationCopy.textContent =
    "Friendly VIP escort is moving through restricted terrain while hostile FPV drone and infantry cells coordinate an ambush.";
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
      isAlternate: decision.is_alternate,
      applicability: decision.applicability,
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
  chooseFriendlyDroneTargets();
  applySelectedDecision();

  for (const unit of state.units.filter((item) => item.health > 0)) {
    if (state.dragging?.id !== unit.id) moveUnitTowardTarget(unit);
  }

  resolveCombat();

  if (missionStatus() !== "running") {
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
  terrainLayer.appendChild(el("rect", {
    x: VIP_EXTRACTION_X,
    y: 0,
    width: WIDTH - VIP_EXTRACTION_X,
    height: HEIGHT,
    fill: "#2ec27e",
    opacity: 0.06,
  }));
  terrainLayer.appendChild(el("line", {
    x1: VIP_EXTRACTION_X,
    y1: 24,
    x2: VIP_EXTRACTION_X,
    y2: HEIGHT - 24,
    stroke: "#8ff0a4",
    "stroke-width": 3,
    "stroke-dasharray": "10 10",
    opacity: 0.55,
  }));
  const label = el("text", {
    x: VIP_EXTRACTION_X + 14,
    y: 52,
    fill: "#b8f7c8",
    "font-size": 18,
    "font-weight": 800,
    "letter-spacing": 2,
  });
  label.textContent = "EXFIL";
  terrainLayer.appendChild(label);
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

function renderSensors() {
  sensorLayer.replaceChildren();
  for (const unit of getFriendlyUnits()) {
    sensorLayer.appendChild(el("circle", {
      cx: unit.x,
      cy: unit.y,
      r: LOCAL_SENSOR_RANGE,
      class: "sensor-ring",
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
      "aria-label": `${unit.side} ${unit.label} ${unitHealthPercent(unit)} percent health`,
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
    type.textContent = displayUnitType(unit);
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
      width: 58 * (unit.health / unit.maxHealth),
      height: 6,
      rx: 3,
      class: `health-bar ${unitHealthPercent(unit) < 40 ? "low" : ""}`,
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
  updateSelectedDecisionTrend(decisions);
  state.renderedDecisions = decisions;
  const signature = decisions
    .map((decision) => `${decision.id}:${Math.round(decision.score)}:${decision.isAlternate ? "alt" : "primary"}`)
    .join("|");
  const now = Date.now();
  if (
    decisionList.childElementCount > 0
    && decisionList.childElementCount === decisions.length
    && signature === state.decisionSignature
    && now - state.lastDecisionRenderAt < 1500
  ) {
    return;
  }

  state.decisionSignature = signature;
  state.lastDecisionRenderAt = now;

  if (decisionList.childElementCount !== decisions.length) {
    decisionList.replaceChildren();
    for (let index = 0; index < decisions.length; index += 1) {
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
        if (!decision || decision.isAlternate) return;
        state.selectedDecision = decision;
        state.selectedTrend = {
          id: decision.id,
          score: Number(decision.score) || 0,
          at: Date.now(),
          slope: 0,
        };
        state.lastDecisionRenderAt = 0;
        currentDecision.textContent = `${decision.title} (${Math.round(decision.score)}% estimated success)`;
        selectedDecisionTrend.textContent = "Selected trend: steady";
        selectedDecisionTrend.classList.remove("up", "down");
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
    card.classList.toggle("alternate", Boolean(decision.isAlternate));
    card.classList.toggle("first-alternate", Boolean(decision.isAlternate) && !decisions[index - 1]?.isAlternate);
    card.querySelector(".decision-title").textContent = decision.title;
    card.querySelector(".score").textContent = `${Math.round(decision.score)}%`;
    card.querySelector(".decision-summary").textContent = decision.summary;
    const selectButton = card.querySelector(".select-decision");
    selectButton.disabled = Boolean(decision.isAlternate);
    selectButton.textContent = decision.isAlternate
      ? "Lower rank"
      : (isSelected ? "Selected" : "Select option");
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
    ["Friendly units", `${getFriendlyUnits().length}/${state.units.filter((unit) => unit.side === "friendly").length}`],
    ["Enemy units", `${getEnemyUnits().length}/${state.units.filter((unit) => unit.side === "enemy").length}`],
    ["Friendly combat power", Math.round(friendlyHealth).toString()],
    ["Enemy combat power", Math.round(enemyHealth).toString()],
    ["Hostile FPV", enemyDrone],
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
  const next = constrainGroundPosition(unit, {
    x: point.x - state.dragging.dx,
    y: point.y - state.dragging.dy,
  });
  unit.x = next.x;
  unit.y = next.y;
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

for (const input of [friendlyCountInput, enemyCountInput]) {
  input.addEventListener("change", () => {
    input.value = String(clamp(Number.parseInt(input.value, 10) || 1, 1, 8));
  });
}

droneAmbushInput?.addEventListener("change", () => {
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

