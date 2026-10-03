import type { Sim } from "./sim";

const BG = "#0e1219";
const GRID = "#1c2433";
const GOLD = "#e4b15a";
const GOLD_DIM = "#8a6a2e";
const CYAN = "#3ec8d8";
const CYAN_DIM = "#1d6c76";
const FG = "#e7e4dc";
const MUTED = "#8d97a8";

export function drawSim(ctx: CanvasRenderingContext2D, sim: Sim, dpr: number, reduced: boolean) {
  const w = sim.w;
  const h = sim.h;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = 40;
  for (let x = step; x < w; x += step) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
  }
  for (let y = step; y < h; y += step) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(w, y + 0.5);
  }
  ctx.stroke();

  if (!reduced) {
    trail(ctx, sim.player.trail, GOLD);
    for (const agent of sim.agents) trail(ctx, agent.body.trail, CYAN);
  }

  cone(ctx, sim.player.x, sim.player.y, sim.player.angle, GOLD, GOLD_DIM);
  sim.agents.forEach((agent, i) => {
    const b = agent.body;
    cone(ctx, b.x, b.y, b.angle, CYAN, CYAN_DIM);
    if (agent.close && sim.aiDrive) {
      ctx.beginPath();
      ctx.arc(b.x, b.y, 26, 0, Math.PI * 2);
      ctx.strokeStyle = CYAN;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.font = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = CYAN;
    ctx.fillText(sim.agents.length > 1 ? String(i + 1) : "net", b.x, b.y + 26);
    ctx.fillText(`${Math.round(agent.dist)} px`, b.x, b.y + 40);
  });

  ctx.font = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillStyle = GOLD;
  ctx.fillText("you", sim.player.x, sim.player.y + 26);

  ctx.textAlign = "left";
  ctx.fillStyle = MUTED;
  ctx.fillText("you", 16, 22);
  ctx.fillStyle = FG;
  ctx.fillText(`${Math.round(sim.player.x)} , ${Math.round(sim.player.y)}`, 16, 38);
}

function trail(ctx: CanvasRenderingContext2D, pts: [number, number][], color: string) {
  for (let i = 0; i < pts.length; i++) {
    const t = (i + 1) / pts.length;
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = t * 0.35;
    ctx.arc(pts[i][0], pts[i][1], 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function cone(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  angle: number,
  fill: string,
  edge: string,
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle);
  ctx.beginPath();
  ctx.moveTo(16, 0);
  ctx.lineTo(-10, 8);
  ctx.lineTo(-6, 0);
  ctx.lineTo(-10, -8);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = edge;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(0, 0, 2.2, 0, Math.PI * 2);
  ctx.fillStyle = BG;
  ctx.fill();
  ctx.restore();
}
