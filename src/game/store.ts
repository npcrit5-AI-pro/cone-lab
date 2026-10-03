/** On-device database. Survives reload and closing the tab. Not shared across devices. */

const DB_NAME = "cone-lab";
const STORE = "brains";
const KEY = "lab";

export type BrainFile = {
  v: 4;
  picked?: boolean;
  agents: { driver: unknown; predictor: unknown }[];
};

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

export async function readBrain(): Promise<BrainFile | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(KEY);
    req.onsuccess = () => {
      const value = req.result;
      resolve(value && typeof value === "object" ? (value as BrainFile) : null);
    };
    req.onerror = () => resolve(null);
  });
}

export async function writeBrain(file: BrainFile): Promise<void> {
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(file, KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}
