import { DRIVE_IN, HIDDEN, MAX_FRAME, PRED_IN, SEE_EVERY, SPEED, TinyNet, type Vec2 } from "./net";
import type { BrainFile } from "./store";

/** Screen position of this viewport on the imaginary desk. */
export const WINDOW_X = 120;
export const WINDOW_Y = 80;
export const MAX_AGENTS = 8;

/** Top-down strafe, not vehicle yaw. A/← decreases x (left on screen). D/→ increases x. */
const SAVE_KEY = "cone-lab-weights-v4";
const LEGACY_KEYS = ["cone-lab-weights-v3", "cone-lab-weights-v2"];
const TOUCH = 36;

export type Body = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Last facing. Held while nearly stopped so the cone does not snap upright. */
  angle: number;
  trail: Vec2[];
};

export type Sense = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  at: number;
  /** True once a velocity has been measured from two looks. */
  trust: boolean;
};

export type Agent = {
  driver: TinyNet;
  predictor: TinyNet;
  body: Body;
  placed: boolean;
  close: boolean;
  guess: Vec2;
  dist: number;
  /** Higher means it has been closer lately. Used when two cones mix. */
  near: number;
};

export type ControlsProbe = {
  getYaw: () => number;
  getSpeed: () => number;
  getX: () => number;
  getY: () => number;
  setKeys: (codes: string[]) => void;
};

declare global {
  interface Window {
    __controlsTest?: ControlsProbe;
  }
}

export class Sim {
  agents: Agent[] = [];
  player: Body = body();
  pointer: Vec2 = [0, 0];
  w = 0;
  h = 0;
  training = true;
  aiDrive = true;
  paused = true;
  started = false;
  keys = new Set<string>();
  /** When set, replaces the keyboard until the next real keydown. */
  injected: string[] | null = null;
  stickX = 0;
  stickY = 0;
  lossHist: number[] = [];
  locked = false;
  placed = false;
  inspect = 0;
  /** Which diagram is open for names. Null hides the list. */
  detail: "predict" | "steer" | null = null;
  /** True once the roster size was chosen, so a reload does not force four again. */
  rosterPicked = false;
  time = 0;
  nextSee = 0;
  sense: Sense | null = null;
  evoNote = "";
  evoUntil = 0;
  private nextEvo = 18;
  private rng = mulberry32(11);

  constructor() {
    this.agents = [1, 2, 3, 4].map((n) => makeAgent(n * 17 + 3));
    this.restore();
    this.ensureRoster();
  }

  resize(w: number, h: number) {
    this.w = w;
    this.h = h;
    if (!this.placed && w > 8 && h > 8) {
      this.player.x = w * 0.5;
      this.player.y = h * 0.55;
      this.pointer = [this.player.x, this.player.y];
      this.spreadUnplaced();
      this.placed = true;
    } else {
      this.clampBody(this.player);
      for (const agent of this.agents) this.clampBody(agent.body);
      this.pointer = [clamp(this.pointer[0], 0, w), clamp(this.pointer[1], 0, h)];
    }
  }

  setPointer(x: number, y: number) {
    if (this.w < 2) return;
    this.pointer = [clamp(x, 0, this.w), clamp(y, 0, this.h)];
  }

  /** Start another cone with its own weights. Saved with the rest. Returns the new count. */
  addAgent() {
    if (this.agents.length >= MAX_AGENTS) return this.agents.length;
    const agent = makeAgent((Math.random() * 10_000) | 0);
    this.placeFresh(agent);
    this.rosterPicked = true;
    this.agents.push(agent);
    this.inspect = this.agents.length - 1;
    this.persist();
    return this.agents.length;
  }

  /** Drop the last cone. Keeps at least one. */
  removeAgent() {
    if (this.agents.length <= 1) return this.agents.length;
    this.rosterPicked = true;
    this.agents.pop();
    if (this.inspect >= this.agents.length) this.inspect = this.agents.length - 1;
    this.persist();
    return this.agents.length;
  }

  totalSteps() {
    let n = 0;
    for (const agent of this.agents) n += agent.driver.steps + agent.predictor.steps;
    return n;
  }

  /** px/s. Heading is the held facing. */
  probe(): ControlsProbe {
    return {
      getYaw: () => this.player.angle,
      getSpeed: () => Math.hypot(this.player.vx, this.player.vy) * 60,
      getX: () => this.player.x,
      getY: () => this.player.y,
      setKeys: (codes: string[]) => {
        this.injected = codes;
      },
    };
  }

  noteKeyDown() {
    this.injected = null;
  }

  step(dt: number) {
    if (!this.started || this.paused || this.w < 8 || this.h < 8) return;

    let ix = this.stickX;
    let iy = this.stickY;
    const held = new Set(this.injected ?? this.keys);
    if (held.has("KeyA") || held.has("ArrowLeft")) ix -= 1;
    if (held.has("KeyD") || held.has("ArrowRight")) ix += 1;
    if (held.has("KeyW") || held.has("ArrowUp")) iy -= 1;
    if (held.has("KeyS") || held.has("ArrowDown")) iy += 1;
    const mag = Math.hypot(ix, iy);
    const nx = mag > 1 ? ix / mag : ix;
    const ny = mag > 1 ? iy / mag : iy;

    const prevX = this.player.x;
    const prevY = this.player.y;
    if (mag > 0.05) {
      this.player.vx = nx * MAX_FRAME;
      this.player.vy = ny * MAX_FRAME;
      this.player.x += this.player.vx * 60 * dt;
      this.player.y += this.player.vy * 60 * dt;
    } else {
      const dx = this.pointer[0] - this.player.x;
      const dy = this.pointer[1] - this.player.y;
      const dist = Math.hypot(dx, dy);
      const gain = 1 - Math.exp(-36 * dt);
      if (dist > 0.4) {
        this.player.x += dx * gain;
        this.player.y += dy * gain;
      }
      const scale = 1 / Math.max(dt, 1e-4);
      this.player.vx = ((this.player.x - prevX) * scale) / 60;
      this.player.vy = ((this.player.y - prevY) * scale) / 60;
    }
    this.clampBody(this.player);
    face(this.player);
    pushTrail(this.player);

    this.time += dt;
    if (this.time + 1e-4 >= this.nextSee) {
      this.observe();
      this.nextSee = this.time + SEE_EVERY;
    }

    const sense = this.sense;
    const span = Math.hypot(this.w, this.h) || 1;
    let nearest = Infinity;
    let anyLock = false;
    let lockedEdge = false;
    let lossSum = 0;

    let bestI = 0;
    for (const agent of this.agents) {
      const ax = agent.body.x;
      const ay = agent.body.y;
      const live = Math.hypot(this.player.x - ax, this.player.y - ay);
      const touch = live < TOUCH;
      const tau = sense ? clamp(this.time - sense.at, 0, SEE_EVERY + 0.25) : 0;
      const coastX = sense ? sense.x + sense.vx * tau : this.player.x;
      const coastY = sense ? sense.y + sense.vy * tau : this.player.y;
      const coastRange = Math.hypot(coastX - ax, coastY - ay);
      const seen = [
        sense ? clamp(sense.vx / SPEED, -1.5, 1.5) : 0,
        sense ? clamp(sense.vy / SPEED, -1.5, 1.5) : 0,
        tau / SEE_EVERY,
        clamp(live / span, 0, 1.5),
        touch ? 1 : 0,
        clamp((live - coastRange) / span, -1.5, 1.5),
      ];
      const shift = agent.predictor.forward(seen);
      const belief = reviseBelief(ax, ay, coastX, coastY, shift[0] * SPEED, shift[1] * SPEED, live, touch);
      agent.guess = belief;
      if (this.training && sense?.trust) {
        agent.predictor.learn(
          seen,
          [clamp((this.player.x - coastX) / SPEED, -2, 2), clamp((this.player.y - coastY) / SPEED, -2, 2)],
          0.05,
        );
      }

      const driveX = [
        (belief[0] - ax) / Math.max(this.w, 1),
        (belief[1] - ay) / Math.max(this.h, 1),
        clamp(live / span, 0, 1.5),
        touch ? 1 : 0,
      ];
      const mean = agent.driver.forward(driveX);
      const m0 = mean[0];
      const m1 = mean[1];
      const skilled = clamp(agent.driver.steps / 2500, 0, 1);
      let jx = 0;
      let jy = 0;
      if (this.training && this.aiDrive && !touch) {
        const base = live > 80 ? 1.15 : live > 40 ? 0.4 : 0.08;
        const sigma = base * (1 - 0.7 * skilled);
        jx = gauss(this.rng) * sigma;
        jy = gauss(this.rng) * sigma;
      }
      const steerX = clamp(m0 + jx, -MAX_FRAME, MAX_FRAME);
      const steerY = clamp(m1 + jy, -MAX_FRAME, MAX_FRAME);
      if (this.aiDrive) {
        agent.body.vx = touch ? steerX * 0.25 : steerX;
        agent.body.vy = touch ? steerY * 0.25 : steerY;
        agent.body.x += agent.body.vx * 60 * dt;
        agent.body.y += agent.body.vy * 60 * dt;
        this.clampBody(agent.body);
      } else {
        agent.body.vx = 0;
        agent.body.vy = 0;
      }
      face(agent.body);
      pushTrail(agent.body);

      const trueDist = Math.hypot(this.player.x - agent.body.x, this.player.y - agent.body.y);
      agent.dist = trueDist;
      agent.near += (1 / (1 + trueDist) - agent.near) * 0.02;
      if (trueDist < nearest) {
        nearest = trueDist;
        bestI = this.agents.indexOf(agent);
      }
      const close = trueDist < TOUCH;
      if (close && !agent.close) lockedEdge = true;
      agent.close = close;
      if (close) anyLock = true;
      if (this.training && this.aiDrive) {
        if (touch) {
          agent.driver.learn(driveX, [0, 0], 0.12);
        } else {
          const adv = clamp((live - trueDist) / 2.2, -1, 1);
          if (Math.abs(adv) > 0.05) {
            const lr = 0.18 * (1 - 0.45 * skilled);
            agent.driver.learn(
              driveX,
              [clamp(m0 + adv * jx, -MAX_FRAME, MAX_FRAME), clamp(m1 + adv * jy, -MAX_FRAME, MAX_FRAME)],
              lr,
            );
          }
        }
      }
      lossSum += agent.driver.loss + agent.predictor.loss;
    }

    this.inspect = this.detail ? this.inspect : bestI;
    this.locked = anyLock;
    if (this.time >= this.nextEvo) {
      this.evolve();
      this.nextEvo = this.time + 16 + this.rng() * 12;
    }
    if (this.lossHist.length > 180) this.lossHist.shift();
    this.lossHist.push(lossSum / this.agents.length);
    return { lockedEdge, dist: nearest };
  }

  /** How long until the next real look at the player, in seconds. */
  seeIn() {
    return Math.max(0, this.nextSee - this.time);
  }

  snapshot(): BrainFile {
    return {
      v: 4,
      picked: this.rosterPicked,
      agents: this.agents.map((agent) => ({
        driver: agent.driver.toJSON(),
        predictor: agent.predictor.toJSON(),
      })),
    };
  }

  persist() {
    const file = this.snapshot();
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(file));
    } catch {
      /* private mode */
    }
    return file;
  }

  /** Keep whichever brain has learned more. Boot replaces the roster; later loads only copy weights. */
  ingest(data: unknown) {
    const parsed = parseAgents(data);
    if (!parsed.length) return false;
    const incoming = parsed.reduce((sum, agent) => sum + agent.driver.steps + agent.predictor.steps, 0);
    if (incoming < this.totalSteps()) return false;
    const picked = data && typeof data === "object" && (data as { picked?: unknown }).picked === true;
    if (!this.started) this.rosterPicked = picked === true;
    if (!this.started) {
      this.agents = parsed;
      this.inspect = clamp(this.inspect, 0, parsed.length - 1);
      if (this.placed) this.spreadUnplaced();
      return true;
    }
    const n = Math.min(this.agents.length, parsed.length);
    for (let i = 0; i < n; i++) {
      this.agents[i].driver.load(parsed[i].driver.toJSON());
      this.agents[i].predictor.load(parsed[i].predictor.toJSON());
    }
    for (let i = this.agents.length; i < parsed.length && this.agents.length < MAX_AGENTS; i++) {
      this.placeFresh(parsed[i]);
      this.agents.push(parsed[i]);
    }
    return true;
  }

  /** A new lab has four cones. A saved roster that already chose a size is left alone. */
  ensureRoster() {
    if (this.rosterPicked) return;
    while (this.agents.length < 4) {
      const agent = makeAgent(80 + this.agents.length * 19);
      this.placeFresh(agent);
      this.agents.push(agent);
    }
    this.rosterPicked = true;
  }

  private evolve() {
    const notes: string[] = [];
    if (this.agents.length >= 2 && this.rng() < 0.7) {
      let i = (this.rng() * this.agents.length) | 0;
      let j = (this.rng() * (this.agents.length - 1)) | 0;
      if (j >= i) j += 1;
      const worse = this.agents[i].near <= this.agents[j].near ? i : j;
      const childD = new TinyNet(DRIVE_IN, HIDDEN, 2, 1, 0.35);
      const childP = new TinyNet(PRED_IN, HIDDEN, 2, 2, 0.05);
      childD.cross(this.agents[i].driver, this.agents[j].driver, this.rng);
      childP.cross(this.agents[i].predictor, this.agents[j].predictor, this.rng);
      childD.mutate(this.rng, 0.06, 0.04);
      childP.mutate(this.rng, 0.06, 0.03);
      this.agents[worse].driver = childD;
      this.agents[worse].predictor = childP;
      this.agents[worse].near *= 0.5;
      notes.push(`mixed ${Math.min(i, j) + 1}+${Math.max(i, j) + 1}`);
    }
    if (this.agents.length === 1 || this.rng() < 0.75) {
      const k = (this.rng() * this.agents.length) | 0;
      this.agents[k].driver.mutate(this.rng, 0.16, 0.1);
      this.agents[k].predictor.mutate(this.rng, 0.16, 0.07);
      notes.push(`mutated ${k + 1}`);
    }
    if (notes.length) {
      this.evoNote = notes.join(" · ");
      this.evoUntil = this.time + 4;
    }
  }

  private observe() {
    const x = this.player.x;
    const y = this.player.y;
    const prev = this.sense;
    let vx = 0;
    let vy = 0;
    let trust = false;
    if (prev) {
      const dt = Math.max(0.05, this.time - prev.at);
      vx = clamp((x - prev.x) / dt, -700, 700);
      vy = clamp((y - prev.y) / dt, -700, 700);
      trust = true;
    }
    this.sense = { x, y, vx, vy, at: this.time, trust };
  }

  private restore() {
    try {
      const raw = localStorage.getItem(SAVE_KEY);
      if (raw && this.ingest(JSON.parse(raw))) return;
      for (const key of LEGACY_KEYS) {
        const legacy = localStorage.getItem(key);
        if (legacy && this.ingest(JSON.parse(legacy))) return;
      }
    } catch {
      /* keep the fresh net */
    }
  }

  private placeFresh(agent: Agent) {
    const ang = this.rng() * Math.PI * 2;
    const reach = Math.min(this.w || 400, this.h || 300) * 0.28;
    agent.body.x = clamp(this.player.x + Math.cos(ang) * reach, 18, Math.max(18, (this.w || 400) - 18));
    agent.body.y = clamp(this.player.y + Math.sin(ang) * reach, 18, Math.max(18, (this.h || 300) - 18));
    agent.placed = this.w > 8;
  }

  private spreadUnplaced() {
    const n = this.agents.length;
    for (let i = 0; i < n; i++) {
      const agent = this.agents[i];
      if (agent.placed) continue;
      const ang = (i / n) * Math.PI * 2 - Math.PI / 2;
      agent.body.x = clamp(this.w * 0.5 + Math.cos(ang) * this.w * 0.26, 18, this.w - 18);
      agent.body.y = clamp(this.h * 0.48 + Math.sin(ang) * this.h * 0.28, 18, this.h - 18);
      agent.placed = true;
    }
  }

  private clampBody(b: Body) {
    const m = 18;
    b.x = clamp(b.x, m, Math.max(m, this.w - m));
    b.y = clamp(b.y, m, Math.max(m, this.h - m));
  }
}

function makeAgent(seed: number): Agent {
  return {
    driver: new TinyNet(DRIVE_IN, HIDDEN, 2, seed, 0.35),
    predictor: new TinyNet(PRED_IN, HIDDEN, 2, seed + 17, 0.05),
    body: body(),
    placed: false,
    close: false,
    guess: [0, 0],
    dist: 0,
    near: 0,
  };
}

function parseAgents(data: unknown): Agent[] {
  if (!data || typeof data !== "object") return [];
  const rec = data as { agents?: unknown; w1?: unknown };
  if (Array.isArray(rec.w1)) return parseOneList([data]);
  if (!Array.isArray(rec.agents)) return [];
  return parseOneList(rec.agents);
}

function parseOneList(list: unknown[]): Agent[] {
  const out: Agent[] = [];
  for (const item of list.slice(0, MAX_AGENTS)) {
    const driver = new TinyNet(DRIVE_IN, HIDDEN, 2, 1, 0.35);
    const predictor = new TinyNet(PRED_IN, HIDDEN, 2, 2, 0.05);
    if (item && typeof item === "object" && "driver" in item) {
      const rec = item as { driver?: unknown; predictor?: unknown };
      if (!driver.load(rec.driver)) continue;
      predictor.load(rec.predictor);
    } else if (!driver.load(item)) {
      continue;
    }
    out.push({
      driver,
      predictor,
      body: body(),
      placed: false,
      close: false,
      guess: [0, 0],
      dist: 0,
      near: 0,
    });
  }
  return out;
}

function reviseBelief(
  ax: number,
  ay: number,
  coastX: number,
  coastY: number,
  shiftX: number,
  shiftY: number,
  dist: number,
  touch: boolean,
): Vec2 {
  if (touch) return [ax, ay];
  let dx = coastX + shiftX - ax;
  let dy = coastY + shiftY - ay;
  let len = Math.hypot(dx, dy);
  if (len < 1) {
    dx = coastX - ax;
    dy = coastY - ay;
    len = Math.hypot(dx, dy);
  }
  if (len < 1) return [ax + dist, ay];
  return [ax + (dx / len) * dist, ay + (dy / len) * dist];
}

function body(): Body {
  return { x: 0, y: 0, vx: 0, vy: 0, angle: -Math.PI / 2, trail: [] };
}

function face(b: Body) {
  if (Math.hypot(b.vx, b.vy) > 0.35) b.angle = Math.atan2(b.vy, b.vx);
}

function pushTrail(b: Body) {
  b.trail.push([b.x, b.y]);
  if (b.trail.length > 16) b.trail.shift();
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}
