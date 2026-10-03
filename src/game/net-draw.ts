import type { TinyNet } from "./net";
import type { Sim } from "./sim";

const BG = "#0e1219";
const GOLD = "#e4b15a";
const CYAN = "#3ec8d8";
const FG = "#e7e4dc";
const MUTED = "#8d97a8";
const GRID = "#1c2433";

const PRED_IN: [string, string][] = [
  ["vx", "Last sideways speed. It only updates every two seconds."],
  ["vy", "Last vertical speed. It only updates every two seconds."],
  ["t", "How long since that look."],
  ["d", "How far you are, all the time."],
  ["hit", "On when this cone is touching you."],
  ["err", "How wrong the straight-line distance is."],
];
const PRED_OUT: [string, string][] = [
  ["dx", "Pushes its idea of you sideways."],
  ["dy", "Pushes its idea of you up or down."],
];
const DRIVE_IN: [string, string][] = [
  ["x", "Sideways toward where it thinks you are."],
  ["y", "Up or down toward where it thinks you are."],
  ["d", "How far you are, all the time."],
  ["hit", "On when this cone is touching you."],
];
const DRIVE_OUT: [string, string][] = [
  ["vx", "How hard it moves sideways."],
  ["vy", "How hard it moves up or down."],
];

export type NeuronRow = { name: string; does: string; state: string };

export function neuronRows(net: TinyNet, kind: "predict" | "steer"): NeuronRow[] {
  const inputs = kind === "predict" ? PRED_IN : DRIVE_IN;
  const outputs = kind === "predict" ? PRED_OUT : DRIVE_OUT;
  const rows: NeuronRow[] = [];
  for (let i = 0; i < net.nIn; i++) {
    const row = inputs[i];
    rows.push({ name: row?.[0] ?? `in ${i + 1}`, does: row?.[1] ?? "Input.", state: heat(1) });
  }
  for (let j = 0; j < net.nHid; j++) {
    rows.push({
      name: `h${j + 1}`,
      does: lean(net.w1, j * net.nIn, net.nIn, inputs, "in"),
      state: heat(net.hidden[j] ?? 0),
    });
  }
  const hid = Array.from({ length: net.nHid }, (_, j) => [`h${j + 1}`, ""] as [string, string]);
  for (let m = 0; m < net.nReason; m++) {
    rows.push({
      name: `r${m + 1}`,
      does: `Reasoning step. ${lean(net.wr, m * net.nHid, net.nHid, hid, "h")}`,
      state: heat(net.reason[m] ?? 0),
    });
  }
  for (let k = 0; k < net.nOut; k++) {
    const row = outputs[k];
    rows.push({ name: row?.[0] ?? `out ${k + 1}`, does: row?.[1] ?? "Output.", state: "" });
  }
  return rows;
}

/** Vertical gap between neurons. Used to keep labels from landing on each other. */
export function columnGap(top: number, bottom: number, n: number) {
  if (n <= 1) return Math.max(0, bottom - top);
  return (bottom - top) / (n - 1);
}

export function fillNames(el: HTMLElement, title: string, rows: NeuronRow[]) {
  const sig = title + "\n" + rows.map((row) => row.name + row.does).join("\n");
  const host = el as HTMLElement & { _sig?: string; _slots?: { does: HTMLElement; state: HTMLElement }[] };
  if (host._sig !== sig || !host._slots || host._slots.length !== rows.length) {
    el.replaceChildren();
    const head = document.createElement("p");
    head.className = "mb-1 font-sans text-sm text-fg";
    head.textContent = title;
    el.appendChild(head);
    const slots: { does: HTMLElement; state: HTMLElement }[] = [];
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "grid grid-cols-[4.2rem_1fr_5.5rem] items-baseline gap-x-4 border-t border-border py-3";
      const name = document.createElement("span");
      name.className = "font-mono text-cyan";
      name.textContent = row.name;
      const does = document.createElement("span");
      does.className = "font-sans text-muted";
      does.textContent = row.does;
      const state = document.createElement("span");
      state.className = "text-right font-mono text-fg";
      state.textContent = row.state;
      line.append(name, does, state);
      el.appendChild(line);
      slots.push({ does, state });
    }
    host._sig = sig;
    host._slots = slots;
    return;
  }
  for (let i = 0; i < rows.length; i++) {
    const slot = host._slots[i];
    if (!slot) continue;
    if (slot.does.textContent !== rows[i].does) slot.does.textContent = rows[i].does;
    if (slot.state.textContent !== rows[i].state) slot.state.textContent = rows[i].state;
  }
}

function lean(w: Float64Array, start: number, n: number, names: [string, string][], fallback: string) {
  let best = 0;
  let mag = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.abs(w[start + i] ?? 0);
    if (a > mag) {
      mag = a;
      best = i;
    }
  }
  if (mag < 0.02) return "Quiet. No strong input yet.";
  const dir = (w[start + best] ?? 0) >= 0 ? "Follows" : "Opposes";
  return `${dir} ${names[best]?.[0] ?? fallback + (best + 1)}.`;
}

function heat(v: number) {
  if (Math.abs(v) < 0.15) return "quiet";
  return v > 0 ? "firing" : "held back";
}

export function drawNet(ctx: CanvasRenderingContext2D, sim: Sim, w: number, h: number, dpr: number) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, w, h);

  const agent = sim.agents[sim.inspect] ?? sim.agents[0];
  if (!agent || w < 40 || h < 40) return;

  ctx.font = "500 12px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = MUTED;

  if (sim.detail) {
    const kind = sim.detail;
    const net = kind === "predict" ? agent.predictor : agent.driver;
    ctx.fillStyle = FG;
    ctx.fillText(`${kind}  ·  cone ${sim.inspect + 1}`, 16, 22);
    ctx.fillStyle = MUTED;
    ctx.fillText("other half switches  ·  same half closes", w > 520 ? 220 : 16, w > 520 ? 22 : 40);
    const labels = kind === "predict" ? ["vx", "vy", "t", "d", "hit", "err"] : ["x", "y", "d", "hit"];
    const outs = kind === "predict" ? ["dx", "dy"] : ["vx", "vy"];
    const top = w > 520 ? 44 : 56;
    drawOne(ctx, net, 16, w - 16, top, h - 18, labels, outs, true);
    return;
  }

  const mid = w * 0.5;
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mid + 0.5, 8);
  ctx.lineTo(mid + 0.5, h - 8);
  ctx.stroke();
  ctx.fillStyle = MUTED;
  ctx.fillText("predict", 12, 18);
  ctx.fillText("steer", mid + 12, 18);
  drawOne(ctx, agent.predictor, 8, mid - 8, 32, h - 12, ["vx", "vy", "t", "d", "hit", "err"], ["dx", "dy"], false);
  drawOne(ctx, agent.driver, mid + 8, w - 8, 32, h - 12, ["x", "y", "d", "hit"], ["vx", "vy"], false);
}

function drawOne(
  ctx: CanvasRenderingContext2D,
  net: TinyNet,
  left: number,
  right: number,
  top: number,
  bottom: number,
  inLabels: string[],
  outLabels: string[],
  named: boolean,
) {
  const span = Math.max(40, right - left);
  const xs = (named ? [0.24, 0.44, 0.64, 0.8] : [0.16, 0.42, 0.66, 0.88]).map((t) => left + span * t);
  const inputs = column(xs[0], top, bottom, net.nIn);
  const hidden = column(xs[1], top, bottom, net.nHid);
  const reason = column(xs[2], top, bottom, net.nReason);
  const outputs = column(xs[3], top, bottom, net.nOut);

  let peak = 0.15;
  for (const weights of [net.w1, net.wr, net.w2]) {
    for (let i = 0; i < weights.length; i++) peak = Math.max(peak, Math.abs(weights[i]));
  }

  for (let j = 0; j < net.nHid; j++) {
    for (let i = 0; i < net.nIn; i++) link(ctx, inputs[i], hidden[j], net.w1[j * net.nIn + i], peak);
  }
  for (let m = 0; m < net.nReason; m++) {
    for (let j = 0; j < net.nHid; j++) link(ctx, hidden[j], reason[m], net.wr[m * net.nHid + j], peak);
  }
  for (let k = 0; k < net.nOut; k++) {
    for (let m = 0; m < net.nReason; m++) link(ctx, reason[m], outputs[k], net.w2[k * net.nReason + m], peak);
  }

  inputs.forEach((p, i) => {
    node(ctx, p[0], p[1], 1);
    if (named) tag(ctx, p[0], p[1], inLabels[i] ?? "", "left");
  });
  hidden.forEach((p, j) => {
    node(ctx, p[0], p[1], net.hidden[j] ?? 0);
    if (named) tag(ctx, p[0], p[1], `h${j + 1}`, "left");
  });
  reason.forEach((p, m) => {
    node(ctx, p[0], p[1], net.reason[m] ?? 0);
    if (named) tag(ctx, p[0], p[1], `r${m + 1}`, "left");
  });
  outputs.forEach((p, k) => {
    node(ctx, p[0], p[1], Math.tanh((net.out[k] ?? 0) / 2));
    if (named) tag(ctx, p[0], p[1], outLabels[k] ?? "", "right");
  });
}

function column(x: number, top: number, bottom: number, n: number): [number, number][] {
  const pts: [number, number][] = [];
  const gap = columnGap(top, bottom, n);
  for (let i = 0; i < n; i++) pts.push([x, n === 1 ? (top + bottom) / 2 : top + i * gap]);
  return pts;
}

function link(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number], w: number, peak: number) {
  const mag = Math.min(1, Math.abs(w) / peak);
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.strokeStyle = w >= 0 ? CYAN : GOLD;
  ctx.globalAlpha = 0.15 + mag * 0.75;
  ctx.lineWidth = 0.6 + mag * 2.2;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function node(ctx: CanvasRenderingContext2D, x: number, y: number, act: number) {
  const on = Math.max(-1, Math.min(1, act));
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fillStyle = on >= 0 ? CYAN : GOLD;
  ctx.globalAlpha = 0.25 + Math.abs(on) * 0.75;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;
  ctx.strokeStyle = FG;
  ctx.stroke();
}

function tag(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, side: "left" | "right") {
  ctx.font = "500 13px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textAlign = side === "left" ? "right" : "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = FG;
  ctx.fillText(text, side === "left" ? x - 14 : x + 14, y);
  ctx.textBaseline = "alphabetic";
}
