import { addPlayers, listPlayers } from './storeService.js';
import { computeBasePrice } from '../utils/basePrice.js';
import { createId } from '../utils/id.js';

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',').map((h) => h.trim());
  const rows = [];
  for (const line of lines) {
    const values = line.split(',').map((v) => v.trim());
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index];
    });
    rows.push(row);
  }
  return rows;
}

function filterGoldRareMen(data) {
  return data.filter((row) => String(row.rarity).toLowerCase() === 'gold rare' && (!row.gender || String(row.gender).toLowerCase() === 'men'));
}

export function importPlayers(payload) {
  let records = [];
  if (Array.isArray(payload)) {
    records = payload;
  } else if (typeof payload === 'string') {
    records = parseCsv(payload);
  } else if (payload && payload.data && typeof payload.data === 'string') {
    records = parseCsv(payload.data);
  }
  const filtered = filterGoldRareMen(records);
  const players = filtered.map((row) => {
    const rating = Number(row.rating);
    return {
      id: createId(),
      name: row.name,
      position: row.position,
      club: row.club,
      nation: row.nation,
      rating,
      rarity: row.rarity,
      basePrice: computeBasePrice(rating)
    };
  });
  addPlayers(players);
  return players;
}

export function getPlayers() {
  return listPlayers();
}
