import { store } from '../data/store.js';
import { createId } from '../utils/id.js';

export function getBasePriceTiers() {
  return store.snapshot.basePriceTiers;
}

export function setBasePriceTiers(tiers) {
  const snapshot = store.snapshot;
  snapshot.basePriceTiers = tiers;
  store.save();
}

export function listPlayers() {
  return store.snapshot.players;
}

export function addPlayers(players) {
  const snapshot = store.snapshot;
  for (const player of players) {
    snapshot.players.push(player);
  }
  store.save();
}

export function findPlayerById(id) {
  return store.snapshot.players.find((p) => p.id === id);
}

export function saveUser(user) {
  const snapshot = store.snapshot;
  snapshot.users.push(user);
  store.save();
}

export function findUserByUsername(username) {
  return store.snapshot.users.find((u) => u.username === username);
}

export function findUserById(id) {
  return store.snapshot.users.find((u) => u.id === id);
}

export function createAuction(data) {
  const snapshot = store.snapshot;
  const auction = { id: createId(), createdAt: new Date().toISOString(), status: 'DRAFT', ...data };
  snapshot.auctions.push(auction);
  store.save();
  return auction;
}

export function getAuction(id) {
  return store.snapshot.auctions.find((a) => a.id === id);
}

export function updateAuction(id, updates) {
  const snapshot = store.snapshot;
  const auction = snapshot.auctions.find((a) => a.id === id);
  if (!auction) return null;
  Object.assign(auction, updates);
  store.save();
  return auction;
}

export function saveAuditLog(entry) {
  const snapshot = store.snapshot;
  snapshot.auditLogs.push({ id: createId(), createdAt: new Date().toISOString(), ...entry });
  store.save();
}

export function listAuditLogsForAuction(auctionId) {
  return store.snapshot.auditLogs.filter((log) => log.auctionId === auctionId);
}

export function createTeam(team) {
  const snapshot = store.snapshot;
  snapshot.teams.push(team);
  store.save();
}

export function replaceTeamsForAuction(auctionId, teams) {
  const snapshot = store.snapshot;
  snapshot.teams = snapshot.teams.filter((team) => team.auctionId !== auctionId);
  for (const team of teams) {
    snapshot.teams.push(team);
  }
  store.save();
}

export function listTeamsByAuction(auctionId) {
  return store.snapshot.teams.filter((team) => team.auctionId === auctionId);
}

export function setTeamBudgets(auctionId, budgets) {
  const snapshot = store.snapshot;
  snapshot.teamBudgets = snapshot.teamBudgets.filter((tb) => tb.auctionId !== auctionId);
  for (const entry of budgets) {
    snapshot.teamBudgets.push(entry);
  }
  store.save();
}

export function getTeamBudget(teamId, auctionId) {
  return store.snapshot.teamBudgets.find((tb) => tb.teamId === teamId && tb.auctionId === auctionId);
}

export function updateTeamBudget(teamId, auctionId, updates) {
  const snapshot = store.snapshot;
  const budget = snapshot.teamBudgets.find((tb) => tb.teamId === teamId && tb.auctionId === auctionId);
  if (!budget) return null;
  Object.assign(budget, updates);
  store.save();
  return budget;
}

export function createLot(lot) {
  const snapshot = store.snapshot;
  snapshot.lots.push(lot);
  store.save();
}

export function listLotsByAuction(auctionId) {
  return store.snapshot.lots.filter((lot) => lot.auctionId === auctionId);
}

export function getActiveLot(auctionId) {
  return store.snapshot.lots.find((lot) => lot.auctionId === auctionId && lot.status === 'ACTIVE');
}

export function updateLot(id, updates) {
  const snapshot = store.snapshot;
  const lot = snapshot.lots.find((l) => l.id === id);
  if (!lot) return null;
  Object.assign(lot, updates);
  store.save();
  return lot;
}

export function addBid(bid) {
  const snapshot = store.snapshot;
  snapshot.bids.push(bid);
  store.save();
}

export function listBidsByLot(lotId) {
  return store.snapshot.bids.filter((b) => b.lotId === lotId);
}

export function createSale(sale) {
  const snapshot = store.snapshot;
  snapshot.sales.push(sale);
  store.save();
}

export function listSalesByAuction(auctionId) {
  const lots = listLotsByAuction(auctionId).map((lot) => lot.id);
  return store.snapshot.sales.filter((sale) => lots.includes(sale.lotId));
}

export function listLotsForTeam(teamId) {
  return store.snapshot.lots.filter((lot) => lot.winnerTeamId === teamId);
}

export function resetAll() {
  store.reset({
    users: [],
    players: [],
    auctions: [],
    teams: [],
    teamBudgets: [],
    lots: [],
    bids: [],
    sales: [],
    auditLogs: [],
    basePriceTiers: store.snapshot.basePriceTiers
  });
}
