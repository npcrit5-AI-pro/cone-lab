# Cone Lab

**Tiny neural nets that chase your cursor and learn on their own.**

Cone Lab is a browser game and a hands-on look at machine learning. You steer a gold cone. A small crowd of cyan cones tries to catch you. Each cyan cone has two tiny neural networks:

- a **predict** net that guesses where you are (they only get a real look at your position and speed every 2 seconds), and
- a **steer** net that decides which way to drive.

Both nets keep learning while you play. Every 16 to 28 seconds the lab also **evolves** them: two cones may *mix* their weights into a child, and one cone may *mutate*. You can watch every neuron light up in the diagram at the bottom of the screen, and the cones' brains are saved in your browser, so they keep getting smarter between visits.

Built with React, TanStack Start, Vite, and an HTML canvas. No server-side AI: all the learning happens in your browser.

---

## Features

- **Play with mouse, keyboard, touch, or the on-screen joystick.** Your cone follows the pointer, WASD / arrow keys, or the stick in the bottom-left.
- **1 to 8 AI cones.** Use **− AI** and **+ AI** to change how many chase you.
- **Live learning.** The **Train** toggle turns learning on or off.
- **Net drives.** The **Net drives** toggle lets the steer nets move the cones, or freezes them in place so you can watch the predict nets guess.
- **Evolution.** Cones periodically mix or mutate. The status line briefly shows notes like `mixed 1+3 · mutated 2`.
- **See inside the brain.** The network diagram shows the predict net (left half) and steer net (right half) of the cone that's closest to you. Click a half to list every neuron by name with its current value.
- **Saved on this device.** Brains are saved every couple of seconds and when you leave the page (IndexedDB plus a `localStorage` backup). Nothing is sent to a server.
- **Sound.** A soft blip when a cone touches you.

---

## Tutorial

### 1. What you need

- **Node.js 22.12 or newer** (the TanStack Start packages require it). Check with `node --version`. Get it from <https://nodejs.org/>.
- **npm** (comes with Node.js)
- **git** (or download the ZIP from GitHub)

### 2. Install

```bash
git clone https://github.com/npcrit5-AI-pro/cone-lab.git
cd cone-lab
npm install
```

### 3. Run

```bash
npm run dev
```

Open **<http://localhost:8080>**. (The dev server listens on all network interfaces on port 8080, so other devices on your network can open `http://<your-computer's-IP>:8080` too.)

Press **Ctrl + C** in the terminal to stop it.

### 4. Play

1. **Start screen.** A card titled **Cone Lab** explains the rules. Click **Enter the lab**, or press **Enter** or **Space**.
2. **Move your cone.** The **gold cone** is you. Move the mouse over the play area, use **W A S D** or the **arrow keys**, or drag the **joystick** in the bottom-left corner (on phones and tablets).
3. **Watch the cyan cones.** They always know how far away you are and whether they're touching you, but they only *see* your exact position and speed every 2 seconds. In between, their predict net guesses where you went.
4. **Read the status line** (bottom panel):
   - `you x, y` is your position, and `dist 1 120px 2 300px ...` is how far each cone is from you.
   - `sees you in 1.4s` counts down to the next real look.
   - The second line shows how many weights each net has (`predict … · steer …`), the total training steps so far, and `saved in this browser`.
5. **Look inside a brain.** The wide diagram under the status line shows the nets of the cone closest to you. **Click the left half** to name every neuron in the *predict* net, or the **right half** for the *steer* net. Click the same half again to close the list.
6. **Use the controls** (buttons under the diagram):

   | Button | Key | What it does |
   |---|---|---|
   | **Train** | `T` | Turn learning on/off. Off = the nets stop changing. |
   | **Net drives** | `F` | On = steer nets drive the cones. Off = cones stay still. |
   | **− AI** / **+ AI (n/8)** | | Remove or add a cyan cone (1 to 8). |
   | **Pause** / **Resume** | `Space` | Freeze or continue the simulation. |

7. **Come back later.** Close the tab whenever you like. Next time, the cones pick up with the brains they had.

### 5. Build for production (optional)

```bash
npm run build      # builds the app
npm run preview    # serves the build at http://127.0.0.1:8081
```

The repo also includes a `vercel.json`, so it can be deployed to [Vercel](https://vercel.com/) as-is.

---

## Configuration

Cone Lab needs **no API keys and no environment variables** to run.

The project comes from the Grok Build app template, which includes optional sign-in and database plumbing that this game does not use. You can ignore these, but for reference:

| Variable | Needed? | What it does |
|---|---|---|
| `VITE_AUTH_ENABLED` | No | Set to `false` to switch the template's (unused) sign-in plumbing fully off. |
| `DATABASE_URL` | No | Postgres URL for build-time migrations. When it's not set, `npm run build` prints `DATABASE_URL not set — skipping` and carries on. |

If you set any, put them in a `.env` file in the project folder (it is git-ignored) and restart `npm run dev`.

Game tuning lives in the code: number of cones (`MAX_AGENTS`), how often cones see you (`SEE_EVERY`), and net sizes (`HIDDEN`, `REASON`) are in `src/game/sim.ts` and `src/game/net.ts`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `npm install` warns `EBADENGINE ... required: { node: '>=22.12.0' }`, or the dev server crashes on start | Your Node.js is too old. Install Node 22.12+ and run `npm install` again. |
| `npm ci` fails with "package.json and package-lock.json are not in sync" | Use `npm install` instead. |
| `Port 8080 is already in use` | Another app is on port 8080. Stop it, or run `npx vite dev --port 3000` and open `http://localhost:3000`. |
| The cones seem dumb again | Brains are stored per browser and per address. A different browser, a private window, or `127.0.0.1` vs `localhost` all start fresh. |
| I want to reset the brains | Clear this site's data in your browser (Settings → Privacy → Site data, or DevTools → Application → Clear storage), then reload. |
| No sound | Sound starts after you click **Enter the lab** (browsers block audio until you interact). Check the tab isn't muted. |
| Fonts look plain | The page loads its fonts from Google Fonts. Offline, it falls back to system fonts. Everything still works. |

### Developer checks

```bash
npm run typecheck   # TypeScript
npm run build       # production build
npm run lint        # ESLint (currently reports a couple of pre-existing warnings/errors)
npm test            # template tests
```

`npm test` runs the Grok Build template's own tests. About 16 of them read files from a `.grok/` folder that only exists in the Grok Build workspace (it's git-ignored), so they fail in a fresh clone. That doesn't affect the game.

---

## Project structure

```
src/
  components/lab.tsx   # Screen layout, controls, joystick, HUD
  game/sim.ts          # Simulation: player, cones, learning, evolution
  game/net.ts          # TinyNet: the small neural network
  game/draw.ts         # Draws the play area
  game/net-draw.ts     # Draws the network diagram and neuron names
  game/store.ts        # Saves brains in IndexedDB
  routes/              # TanStack Start routes (the game is the home page)
scripts/, server/, src/lib/   # Grok Build template plumbing (dev wrapper, PWA, optional auth/db)
```
