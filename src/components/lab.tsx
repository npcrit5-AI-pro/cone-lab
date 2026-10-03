import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Brain, Pause, Play } from "lucide-react";
import { drawSim } from "@/game/draw";
import { drawNet, fillNames, neuronRows } from "@/game/net-draw";
import { MAX_AGENTS, Sim, WINDOW_X, WINDOW_Y } from "@/game/sim";
import { readBrain, writeBrain } from "@/game/store";

const STEP = 1 / 60;

export function Lab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const netRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Sim | null>(null);
  const readRef = useRef<HTMLParagraphElement>(null);
  const metaRef = useRef<HTMLParagraphElement>(null);
  const namesRef = useRef<HTMLDivElement>(null);
  const saveAt = useRef(0);
  const beginRef = useRef<() => void>(() => {});

  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [training, setTraining] = useState(true);
  const [aiDrive, setAiDrive] = useState(true);
  const [aiCount, setAiCount] = useState(4);
  const [detail, setDetail] = useState<"predict" | "steer" | null>(null);
  const [stick, setStick] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const sim = new Sim();
    simRef.current = sim;
    setAiCount(sim.agents.length);
    window.__controlsTest = sim.probe();

    const flush = () => {
      const file = sim.persist();
      void writeBrain(file);
    };
    void readBrain().then((file) => {
      if (!file || simRef.current !== sim) return;
      if (sim.ingest(file)) {
        sim.ensureRoster();
        setAiCount(sim.agents.length);
      }
    });

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const fit = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      sim.resize(rect.width, rect.height);
      drawSim(ctx, sim, dpr, reduced);
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(canvas.parentElement ?? canvas);

    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const audio = { ctx: null as AudioContext | null, cool: 0 };

    const blip = () => {
      const ac = audio.ctx;
      if (!ac || ac.currentTime < audio.cool) return;
      audio.cool = ac.currentTime + 1.4;
      const o = ac.createOscillator();
      const g = ac.createGain();
      o.type = "sine";
      o.frequency.value = 620;
      g.gain.setValueAtTime(0.025, ac.currentTime);
      g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + 0.1);
      o.connect(g);
      g.connect(ac.destination);
      o.start();
      o.stop(ac.currentTime + 0.11);
    };

    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      acc += dt;
      let guard = 0;
      let edge = false;
      while (acc >= STEP && guard < 3) {
        const result = sim.step(STEP);
        if (result?.lockedEdge) edge = true;
        acc -= STEP;
        guard += 1;
      }
      if (guard === 3) acc = 0;
      if (edge) blip();

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      drawSim(ctx, sim, dpr, reduced);
      const net = netRef.current;
      const nctx = net?.getContext("2d");
      if (net && nctx) {
        const rect = net.getBoundingClientRect();
        const bw = Math.max(1, Math.floor(rect.width * dpr));
        const bh = Math.max(1, Math.floor(rect.height * dpr));
        if (net.width !== bw || net.height !== bh) {
          net.width = bw;
          net.height = bh;
        }
        drawNet(nctx, sim, rect.width, rect.height, dpr);
      }
      const names = namesRef.current;
      if (names && sim.detail) {
        const agent = sim.agents[sim.inspect];
        const brain = sim.detail === "predict" ? agent?.predictor : agent?.driver;
        if (brain) fillNames(names, `cone ${sim.inspect + 1}  ·  ${sim.detail}`, neuronRows(brain, sim.detail));
      }
      paintHud(sim, readRef.current, metaRef.current);

      if (sim.started && now - saveAt.current > 2000) {
        saveAt.current = now;
        flush();
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    const begin = () => {
      sim.started = true;
      sim.paused = false;
      setStarted(true);
      setPaused(false);
      const AC = window.AudioContext;
      if (AC && !audio.ctx) {
        audio.ctx = new AC();
        void audio.ctx.resume();
      }
    };
    beginRef.current = begin;

    const down = (e: KeyboardEvent) => {
      const move = MOVE_CODES.has(e.code);
      if (move) e.preventDefault();
      sim.noteKeyDown();
      sim.keys.add(e.code);
      if (e.repeat) return;
      if (!sim.started && (e.code === "Enter" || e.code === "Space")) {
        e.preventDefault();
        begin();
        return;
      }
      if (!sim.started) return;
      if (e.code === "Space") {
        e.preventDefault();
        sim.paused = !sim.paused;
        setPaused(sim.paused);
      } else if (e.code === "KeyT") {
        sim.training = !sim.training;
        setTraining(sim.training);
      } else if (e.code === "KeyF") {
        sim.aiDrive = !sim.aiDrive;
        setAiDrive(sim.aiDrive);
      }
    };
    const up = (e: KeyboardEvent) => {
      sim.keys.delete(e.code);
    };
    const blur = () => {
      sim.keys.clear();
      sim.stickX = 0;
      sim.stickY = 0;
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    window.addEventListener("pagehide", flush);
    const hide = () => {
      if (document.visibilityState === "hidden") flush();
    };
    document.addEventListener("visibilitychange", hide);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", hide);
      if (window.__controlsTest && simRef.current === sim) delete window.__controlsTest;
    };
  }, []);

  function enter() {
    beginRef.current();
  }

  function toggle(kind: "training" | "aiDrive" | "paused") {
    const sim = simRef.current;
    if (!sim) return;
    if (kind === "training") {
      sim.training = !sim.training;
      setTraining(sim.training);
    } else if (kind === "aiDrive") {
      sim.aiDrive = !sim.aiDrive;
      setAiDrive(sim.aiDrive);
    } else {
      sim.paused = !sim.paused;
      setPaused(sim.paused);
    }
  }

  function pointFrom(e: ReactPointerEvent) {
    const canvas = canvasRef.current;
    const sim = simRef.current;
    if (!canvas || !sim) return;
    const rect = canvas.getBoundingClientRect();
    sim.setPointer(e.clientX - rect.left, e.clientY - rect.top);
  }

  return (
    <main className="flex h-dvh flex-col bg-bg text-fg">
      <div
        className="relative min-h-0 flex-1"
        onPointerMove={(e) => {
          if (e.pointerType === "mouse" || e.buttons > 0) pointFrom(e);
        }}
        onPointerDown={pointFrom}
      >
        <canvas ref={canvasRef} className="h-full w-full touch-none" />
        <Joystick
          knob={stick}
          onChange={(x, y) => {
            setStick({ x, y });
            const sim = simRef.current;
            if (!sim) return;
            sim.stickX = x;
            sim.stickY = y;
          }}
        />
        {!started && (
          <div className="absolute inset-0 flex items-center justify-center bg-bg/80 p-4 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-lab border border-border bg-surface p-6 shadow-lg">
              <p className="font-mono text-xs tracking-widest text-gold uppercase">two tiny nets</p>
              <h1 className="mt-2 font-sans text-3xl font-semibold tracking-tight">Cone Lab</h1>
              <p className="mt-2 text-base leading-snug text-muted">
                Your cone follows the cursor. The others always know your distance and whether
                they are touching you. Every two seconds they get your position and speed, and
                they keep changing where they think you are.
              </p>
              <ol className="mt-4 space-y-2 text-sm leading-snug">
                <li>
                  <span className="text-gold">Gold cone</span> is you. Move the cursor, or WASD, or the
                  stick.
                </li>
                <li>
                  <span className="text-cyan">Cyan cones</span> each keep their own nets. Three reason neurons
                  sit in the middle. Their guesses stay hidden.
                </li>
                <li>Click predict or steer to name every neuron. − and + change how many cones there are.</li>
                <li>Every so often one mutates, or two mix their better weights. Closing the page keeps it.</li>
              </ol>
              <p className="mt-4 font-mono text-xs text-muted">T train · F net drives · Space pause</p>
              <button
                type="button"
                onClick={enter}
                className="mt-5 h-11 w-full rounded-lab bg-gold font-semibold text-bg"
              >
                Enter the lab
              </button>
            </div>
          </div>
        )}
      </div>

      <footer className="border-t border-border bg-surface px-4 py-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <Brain className="size-4 text-cyan" aria-hidden="true" />
              <h2 className="font-sans text-sm font-semibold tracking-wide">Cone Lab</h2>
              <span className="font-mono text-xs text-muted">
                viewport {WINDOW_X}, {WINDOW_Y}
              </span>
            </div>
            <p ref={readRef} className="mt-1 font-mono text-xs leading-snug text-fg">
              you — · net —
            </p>
            <p ref={metaRef} className="truncate font-mono text-xs text-muted">
              predict · steer · waiting
            </p>
          </div>
        </div>
        <canvas
          ref={netRef}
          onClick={(e) => {
            const sim = simRef.current;
            const el = netRef.current;
            if (!sim || !el) return;
            const rect = el.getBoundingClientRect();
            const side = e.clientX - rect.left < rect.width / 2 ? "predict" : "steer";
            sim.detail = sim.detail === side ? null : side;
            setDetail(sim.detail);
          }}
          className={`mt-3 w-full cursor-pointer rounded-lab border border-border ${detail ? "h-[26rem]" : "h-36"}`}
          aria-label="Neural network. Click a side to name every neuron."
        />
        {detail && (
          <div
            ref={namesRef}
            className="mt-2 max-h-72 overflow-auto rounded-lab border border-border bg-surface-2 px-4 py-3 text-sm"
          />
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          <Toggle on={training} label="Train" onClick={() => toggle("training")} />
          <Toggle on={aiDrive} label="Net drives" onClick={() => toggle("aiDrive")} />
          <button
            type="button"
            onClick={() => {
              const sim = simRef.current;
              if (!sim) return;
              setAiCount(sim.removeAgent());
            }}
            disabled={aiCount <= 1}
            className="h-11 rounded-lab border border-border bg-surface-2 px-3 text-sm text-fg disabled:opacity-40"
          >
            − AI
          </button>
          <button
            type="button"
            onClick={() => {
              const sim = simRef.current;
              if (!sim) return;
              setAiCount(sim.addAgent());
            }}
            disabled={aiCount >= MAX_AGENTS}
            className="h-11 rounded-lab border border-border bg-surface-2 px-3 text-sm text-fg disabled:opacity-40"
          >
            + AI ({aiCount}/{MAX_AGENTS})
          </button>
          <button
            type="button"
            onClick={() => toggle("paused")}
            disabled={!started}
            className="inline-flex h-11 items-center gap-2 rounded-lab border border-border bg-surface-2 px-3 text-sm text-fg disabled:opacity-40"
          >
            {paused ? <Play className="size-4" /> : <Pause className="size-4" />}
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </footer>
    </main>
  );
}

function Toggle({ on, label, onClick }: { on: boolean; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={
        "h-11 rounded-lab px-3 text-sm font-medium " +
        (on ? "bg-cyan text-bg" : "border border-border bg-surface-2 text-fg")
      }
    >
      {label}
    </button>
  );
}

function Joystick({
  knob,
  onChange,
}: {
  knob: { x: number; y: number };
  onChange: (x: number, y: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  function fromEvent(e: ReactPointerEvent) {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const max = r.width / 2;
    let dx = e.clientX - (r.left + r.width / 2);
    let dy = e.clientY - (r.top + r.height / 2);
    const m = Math.hypot(dx, dy) || 1;
    if (m > max) {
      dx = (dx / m) * max;
      dy = (dy / m) * max;
    }
    onChange(dx / max, dy / max);
  }

  return (
    <div
      ref={ref}
      className="stick absolute bottom-4 left-4 size-28 touch-none items-center justify-center rounded-full border border-border bg-surface/80"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        e.stopPropagation();
        fromEvent(e);
      }}
      onPointerMove={(e) => {
        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
        e.stopPropagation();
        fromEvent(e);
      }}
      onPointerUp={(e) => {
        e.stopPropagation();
        onChange(0, 0);
      }}
      onPointerCancel={() => onChange(0, 0)}
    >
      <span
        className="size-11 rounded-full bg-gold"
        style={{ transform: `translate(${knob.x * 36}px, ${knob.y * 36}px)` }}
      />
    </div>
  );
}

const MOVE_CODES = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
]);

function paintHud(sim: Sim, read: HTMLParagraphElement | null, meta: HTMLParagraphElement | null) {
  const watched = sim.agents[sim.inspect] ?? sim.agents[0];
  if (read) {
    const dists = sim.agents.map((agent, i) => `${i + 1} ${Math.round(agent.dist)}px`).join("  ");
    const evo = sim.time < sim.evoUntil && sim.evoNote ? `   ${sim.evoNote}` : "";
    read.textContent = `you ${Math.round(WINDOW_X + sim.player.x)}, ${Math.round(WINDOW_Y + sim.player.y)}   dist ${dists}   sees you in ${sim.seeIn().toFixed(1)}s${evo}`;
  }
  if (meta && watched) {
    const steps = sim.agents.reduce((sum, agent) => sum + agent.driver.steps + agent.predictor.steps, 0);
    meta.textContent = `predict ${watched.predictor.paramCount()} · steer ${watched.driver.paramCount()} · steps ${steps} · saved in this browser`;
  }
}
