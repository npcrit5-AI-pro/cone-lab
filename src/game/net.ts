/** Driver and predictor: input → 6 tanh → 3 tanh (reason) → output. */

export const HIDDEN = 6;
export const REASON = 3;
export const PRED_IN = 6;
export const DRIVE_IN = 4;
export const MAX_FRAME = 4.5;
export const SEE_EVERY = 2;
/** px/s. Predictor residuals are in units of this. */
export const SPEED = 320;

export type Vec2 = [number, number];

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

export class TinyNet {
  readonly nIn: number;
  readonly nHid: number;
  readonly nReason: number;
  readonly nOut: number;
  w1: Float64Array;
  b1: Float64Array;
  wr: Float64Array;
  br: Float64Array;
  w2: Float64Array;
  b2: Float64Array;
  loss = 0;
  steps = 0;
  hidden: Float64Array;
  reason: Float64Array;
  out: Float64Array;
  private scale: number;

  constructor(nIn: number, nHid: number, nOut: number, seed = 7, scale = 0.35, nReason = REASON) {
    this.nIn = nIn;
    this.nHid = nHid;
    this.nReason = nReason;
    this.nOut = nOut;
    this.scale = scale;
    this.w1 = new Float64Array(nIn * nHid);
    this.b1 = new Float64Array(nHid);
    this.wr = new Float64Array(nReason * nHid);
    this.br = new Float64Array(nReason);
    this.w2 = new Float64Array(nOut * nReason);
    this.b2 = new Float64Array(nOut);
    this.hidden = new Float64Array(nHid);
    this.reason = new Float64Array(nReason);
    this.out = new Float64Array(nOut);
    this.reseed(seed);
  }

  reseed(seed: number) {
    const rng = mulberry32(seed);
    for (const w of [this.w1, this.wr, this.w2]) {
      for (let i = 0; i < w.length; i++) w[i] = gauss(rng) * this.scale;
    }
    this.b1.fill(0);
    this.br.fill(0);
    this.b2.fill(0);
    this.loss = 0;
    this.steps = 0;
    this.hidden.fill(0);
    this.reason.fill(0);
    this.out.fill(0);
  }

  paramCount() {
    return this.w1.length + this.b1.length + this.wr.length + this.br.length + this.w2.length + this.b2.length;
  }

  forward(x: ArrayLike<number>) {
    this.eval(x);
    return Array.from(this.out);
  }

  /** Gradient step. Negative lr pushes the output away from target. */
  learn(x: ArrayLike<number>, target: ArrayLike<number>, lr: number) {
    this.eval(x);
    const err = new Float64Array(this.nOut);
    let loss = 0;
    for (let k = 0; k < this.nOut; k++) {
      err[k] = this.out[k] - target[k];
      loss += err[k] * err[k];
    }
    this.loss = 0.5 * loss;
    if (!Number.isFinite(this.loss)) return this.loss;

    const dr = new Float64Array(this.nReason);
    for (let m = 0; m < this.nReason; m++) {
      let back = 0;
      for (let k = 0; k < this.nOut; k++) back += err[k] * this.w2[k * this.nReason + m];
      dr[m] = back * (1 - this.reason[m] * this.reason[m]);
    }
    const dh = new Float64Array(this.nHid);
    for (let j = 0; j < this.nHid; j++) {
      let back = 0;
      for (let m = 0; m < this.nReason; m++) back += dr[m] * this.wr[m * this.nHid + j];
      dh[j] = back * (1 - this.hidden[j] * this.hidden[j]);
    }
    for (let k = 0; k < this.nOut; k++) {
      for (let m = 0; m < this.nReason; m++) {
        this.w2[k * this.nReason + m] -= lr * this.reason[m] * err[k];
      }
      this.b2[k] -= lr * err[k];
    }
    for (let m = 0; m < this.nReason; m++) {
      for (let j = 0; j < this.nHid; j++) {
        this.wr[m * this.nHid + j] -= lr * this.hidden[j] * dr[m];
      }
      this.br[m] -= lr * dr[m];
    }
    for (let j = 0; j < this.nHid; j++) {
      for (let i = 0; i < this.nIn; i++) {
        this.w1[j * this.nIn + i] -= lr * x[i] * dh[j];
      }
      this.b1[j] -= lr * dh[j];
    }
    this.steps += 1;
    return this.loss;
  }

  /** Sprinkle noise on a fraction of the weights. */
  mutate(rng: () => number, chance: number, sigma: number) {
    for (const w of [this.w1, this.b1, this.wr, this.br, this.w2, this.b2]) {
      for (let i = 0; i < w.length; i++) {
        if (rng() < chance) w[i] += gauss(rng) * sigma;
      }
    }
  }

  /** Each weight is taken from one parent or the other, so both can pass a feature on. */
  cross(a: TinyNet, b: TinyNet, rng: () => number) {
    mix(this.w1, a.w1, b.w1, rng);
    mix(this.b1, a.b1, b.b1, rng);
    mix(this.wr, a.wr, b.wr, rng);
    mix(this.br, a.br, b.br, rng);
    mix(this.w2, a.w2, b.w2, rng);
    mix(this.b2, a.b2, b.b2, rng);
    this.steps = Math.round((a.steps + b.steps) / 2);
  }

  toJSON() {
    return {
      v: 3 as const,
      nIn: this.nIn,
      nHid: this.nHid,
      nReason: this.nReason,
      nOut: this.nOut,
      w1: Array.from(this.w1),
      b1: Array.from(this.b1),
      wr: Array.from(this.wr),
      br: Array.from(this.br),
      w2: Array.from(this.w2),
      b2: Array.from(this.b2),
      steps: this.steps,
    };
  }

  load(data: unknown) {
    if (!data || typeof data !== "object") return false;
    const d = data as Record<string, unknown>;
    if (typeof d.nReason === "number") {
      if (d.nHid !== this.nHid || d.nOut !== this.nOut) return false;
      if (typeof d.nIn !== "number" || typeof d.nReason !== "number") return false;
      if (d.nIn === this.nIn && d.nReason === this.nReason) {
        if (
          !arr(d.w1, this.w1) ||
          !arr(d.b1, this.b1) ||
          !arr(d.wr, this.wr) ||
          !arr(d.br, this.br) ||
          !arr(d.w2, this.w2) ||
          !arr(d.b2, this.b2)
        ) {
          return false;
        }
      } else if (!this.loadGrown(d)) {
        return false;
      }
    } else if (!this.loadFlat(d)) {
      return false;
    }
    this.steps = typeof d.steps === "number" && Number.isFinite(d.steps) ? d.steps : 0;
    return true;
  }

  /** Copy an older, smaller net. New inputs and the extra reason neuron stay at zero. */
  private loadGrown(d: Record<string, unknown>) {
    const oldIn = d.nIn;
    const oldR = d.nReason;
    if (typeof oldIn !== "number" || typeof oldR !== "number") return false;
    if (oldIn < 1 || oldIn > this.nIn || oldR < 1 || oldR > this.nReason) return false;
    const oldW1 = new Float64Array(oldIn * this.nHid);
    const oldWr = new Float64Array(oldR * this.nHid);
    const oldBr = new Float64Array(oldR);
    const oldW2 = new Float64Array(this.nOut * oldR);
    if (!arr(d.w1, oldW1) || !arr(d.b1, this.b1) || !arr(d.wr, oldWr) || !arr(d.br, oldBr)) return false;
    if (!arr(d.w2, oldW2) || !arr(d.b2, this.b2)) return false;
    this.w1.fill(0);
    for (let j = 0; j < this.nHid; j++) {
      for (let i = 0; i < oldIn; i++) this.w1[j * this.nIn + i] = oldW1[j * oldIn + i];
    }
    this.wr.fill(0);
    this.br.fill(0);
    for (let m = 0; m < oldR; m++) {
      this.br[m] = oldBr[m];
      for (let j = 0; j < this.nHid; j++) this.wr[m * this.nHid + j] = oldWr[m * this.nHid + j];
    }
    this.w2.fill(0);
    for (let k = 0; k < this.nOut; k++) {
      for (let m = 0; m < oldR; m++) this.w2[k * this.nReason + m] = oldW2[k * oldR + m];
    }
    return true;
  }

  /** Older nets mapped hidden straight to the output. Fold that into the tiny reason layer. */
  private loadFlat(d: Record<string, unknown>) {
    const direct = new Float64Array(this.nOut * this.nHid);
    const directB = new Float64Array(this.nOut);
    if (typeof d.nIn === "number") {
      if (d.nHid !== this.nHid || d.nOut !== this.nOut || d.nIn > this.nIn) return false;
      if (!arr(d.b1, this.b1) || !arr(d.w2, direct) || !arr(d.b2, directB)) return false;
      const oldW = new Float64Array(d.nIn * this.nHid);
      if (!arr(d.w1, oldW)) return false;
      this.w1.fill(0);
      for (let j = 0; j < this.nHid; j++) {
        for (let i = 0; i < d.nIn; i++) this.w1[j * this.nIn + i] = oldW[j * d.nIn + i];
      }
    } else if (!this.loadLegacy(d, direct, directB)) {
      return false;
    }
    this.adoptHead(direct, directB);
    return true;
  }

  private loadLegacy(d: Record<string, unknown>, direct: Float64Array, directB: Float64Array) {
    if (this.nHid !== 6 || this.nOut !== 2 || this.nIn < 2) return false;
    const w1 = d.w1;
    const b1 = d.b1;
    const w2 = d.w2;
    const b2 = d.b2;
    if (!Array.isArray(w1) || !Array.isArray(b1) || !Array.isArray(w2) || !Array.isArray(b2)) return false;
    if (w1.length !== 12 || b1.length !== 6 || w2.length !== 12 || b2.length !== 2) return false;
    this.w1.fill(0);
    for (let j = 0; j < 6; j++) {
      const a = w1[j];
      const b = w1[6 + j];
      const bias = b1[j];
      if (typeof a !== "number" || typeof b !== "number" || typeof bias !== "number") return false;
      this.w1[j * this.nIn] = a;
      this.w1[j * this.nIn + 1] = b;
      this.b1[j] = bias;
    }
    for (let j = 0; j < 6; j++) {
      for (let k = 0; k < 2; k++) {
        const w = w2[j * 2 + k];
        if (typeof w !== "number" || !Number.isFinite(w)) return false;
        direct[k * 6 + j] = w;
      }
    }
    if (typeof b2[0] !== "number" || typeof b2[1] !== "number") return false;
    directB[0] = b2[0];
    directB[1] = b2[1];
    return true;
  }

  /** y ≈ old linear head, passed through a 2-wide tanh squeeze. */
  private adoptHead(direct: Float64Array, directB: Float64Array) {
    const s = 4;
    for (let m = 0; m < this.nReason; m++) {
      for (let j = 0; j < this.nHid; j++) {
        this.wr[m * this.nHid + j] = m < this.nOut ? direct[m * this.nHid + j] / s : 0;
      }
      this.br[m] = m < directB.length ? directB[m] / s : 0;
    }
    this.w2.fill(0);
    this.b2.fill(0);
    const n = Math.min(this.nOut, this.nReason);
    for (let k = 0; k < n; k++) this.w2[k * this.nReason + k] = s;
  }

  private eval(x: ArrayLike<number>) {
    const h = this.hidden;
    for (let j = 0; j < this.nHid; j++) {
      let pre = this.b1[j];
      const row = j * this.nIn;
      for (let i = 0; i < this.nIn; i++) pre += x[i] * this.w1[row + i];
      h[j] = Math.tanh(pre);
    }
    const r = this.reason;
    for (let m = 0; m < this.nReason; m++) {
      let pre = this.br[m];
      const row = m * this.nHid;
      for (let j = 0; j < this.nHid; j++) pre += h[j] * this.wr[row + j];
      r[m] = Math.tanh(pre);
    }
    for (let k = 0; k < this.nOut; k++) {
      let pre = this.b2[k];
      const row = k * this.nReason;
      for (let m = 0; m < this.nReason; m++) pre += r[m] * this.w2[row + m];
      this.out[k] = pre;
    }
  }
}

function mix(dst: Float64Array, a: Float64Array, b: Float64Array, rng: () => number) {
  for (let i = 0; i < dst.length; i++) dst[i] = rng() < 0.5 ? a[i] : b[i];
}

function arr(src: unknown, dst: Float64Array) {
  if (!Array.isArray(src) || src.length !== dst.length) return false;
  for (let i = 0; i < dst.length; i++) {
    const n = src[i];
    if (typeof n !== "number" || !Number.isFinite(n)) return false;
    dst[i] = n;
  }
  return true;
}
