import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const STORE_PATH = join(__dirname, '../../data/store.json');

const DEFAULT_SNAPSHOT = {
  users: [],
  players: [],
  auctions: [],
  teams: [],
  teamBudgets: [],
  lots: [],
  bids: [],
  sales: [],
  auditLogs: [],
  basePriceTiers: [
    { min: 90, max: 99, price: 300000 },
    { min: 87, max: 89, price: 200000 },
    { min: 84, max: 86, price: 120000 },
    { min: 81, max: 83, price: 70000 },
    { min: 78, max: 80, price: 40000 },
    { min: 75, max: 77, price: 20000 },
    { min: 0, max: 74, price: 10000 }
  ]
};

function ensureStoreFile() {
  if (!existsSync(STORE_PATH)) {
    writeFileSync(STORE_PATH, JSON.stringify(DEFAULT_SNAPSHOT, null, 2));
  }
}

let cache = null;

function loadStore() {
  ensureStoreFile();
  if (!cache) {
    const raw = readFileSync(STORE_PATH, 'utf8');
    cache = JSON.parse(raw);
  }
  return cache;
}

function saveStore() {
  if (!cache) return;
  if (process.env.NODE_ENV === 'test') return;
  writeFileSync(STORE_PATH, JSON.stringify(cache, null, 2));
}

function resetStore(snapshot) {
  cache = JSON.parse(JSON.stringify(snapshot));
  saveStore();
}

export const store = {
  get snapshot() {
    return loadStore();
  },
  clone() {
    return JSON.parse(JSON.stringify(loadStore()));
  },
  save: saveStore,
  reset: resetStore
};
