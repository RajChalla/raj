import {
  createAuction,
  getAuction,
  updateAuction,
  listTeamsByAuction,
  setTeamBudgets,
  getTeamBudget,
  updateTeamBudget,
  createLot,
  listLotsByAuction,
  getActiveLot,
  updateLot,
  addBid,
  listBidsByLot,
  createSale,
  listSalesByAuction,
  saveAuditLog,
  findUserByUsername,
  replaceTeamsForAuction
} from './storeService.js';
import { findPlayerById } from './storeService.js';
import { createId } from '../utils/id.js';
import { computeBasePrice } from '../utils/basePrice.js';
import { validateBid } from '../utils/bidRules.js';
import { canCompleteAuction } from '../utils/auctionRules.js';

export function ensureAuction(auctionId) {
  const auction = getAuction(auctionId);
  if (!auction) throw new Error('Auction not found');
  return auction;
}

export function createAuctionWithSettings({ name, increment = 1000, antiSnipeThreshold = 5, antiSnipeExtension = 10 }) {
  return createAuction({ name, increment, antiSnipeThreshold, antiSnipeExtension });
}

export function setTeamsAndBudgets(auctionId, teamsInput, actorId) {
  const auction = ensureAuction(auctionId);
  if (!Array.isArray(teamsInput) || teamsInput.length === 0) {
    throw new Error('Teams payload required');
  }
  const teams = [];
  const budgets = [];
  for (const entry of teamsInput) {
    const teamId = createId();
    let ownerUserId = entry.ownerUserId || null;
    if (!ownerUserId && entry.ownerUsername) {
      const ownerUser = findUserByUsername(entry.ownerUsername);
      if (!ownerUser) {
        throw new Error(`Owner ${entry.ownerUsername} not found`);
      }
      ownerUserId = ownerUser.id;
    }
    const team = {
      id: teamId,
      auctionId,
      name: entry.name,
      ownerUserId
    };
    teams.push(team);
    budgets.push({
      id: createId(),
      auctionId,
      teamId,
      initialBudget: entry.budget,
      remainingBudget: entry.budget
    });
  }
  replaceTeamsForAuction(auctionId, teams);
  setTeamBudgets(auctionId, budgets);
  saveAuditLog({ auctionId, actorId, action: 'SET_BUDGETS', details: { teams: teamsInput } });
  return teams;
}

export function queueLots(auctionId, playerIds) {
  for (const playerId of playerIds) {
    const player = findPlayerById(playerId);
    if (!player) {
      throw new Error(`Player ${playerId} not found`);
    }
    const lot = {
      id: createId(),
      auctionId,
      playerId,
      reserve: player.basePrice ?? computeBasePrice(player.rating),
      status: 'QUEUED',
      queuedAt: new Date().toISOString()
    };
    createLot(lot);
  }
}

export function activateNextLot(auctionId) {
  const lots = listLotsByAuction(auctionId);
  const next = lots.find((lot) => lot.status === 'QUEUED');
  if (!next) return null;
  if (getActiveLot(auctionId)) {
    throw new Error('Another lot already active');
  }
  const now = Date.now();
  const duration = 30; // default 30s
  const lot = updateLot(next.id, {
    status: 'ACTIVE',
    openedAt: new Date(now).toISOString(),
    closesAt: new Date(now + duration * 1000).toISOString(),
    currentHighBid: null,
    currentHighTeamId: null
  });
  return lot;
}

export function placeBid({ auctionId, lotId, teamId, amount, actorId }) {
  const auction = ensureAuction(auctionId);
  const lot = listLotsByAuction(auctionId).find((l) => l.id === lotId);
  if (!lot) throw new Error('Lot not found');
  if (lot.status !== 'ACTIVE') throw new Error('Lot not active');
  const budget = getTeamBudget(teamId, auctionId);
  if (!budget) throw new Error('Budget not found');
  const bids = listBidsByLot(lotId);
  const currentHigh = bids.length ? Math.max(...bids.map((b) => b.amount)) : null;
  const previousLeader = lot.currentHighTeamId;
  const previousAmount = lot.currentHighBid || 0;
  const availableBudget = budget.remainingBudget + (previousLeader === teamId ? previousAmount : 0);
  const error = validateBid({
    amount,
    basePrice: lot.reserve,
    currentHigh,
    remainingBudget: availableBudget,
    increment: auction.increment
  });
  if (error) throw new Error(error);
  const bid = {
    id: createId(),
    auctionId,
    lotId,
    teamId,
    amount,
    placedAt: new Date().toISOString(),
    actorId
  };
  addBid(bid);
  updateLot(lotId, {
    currentHighBid: amount,
    currentHighTeamId: teamId
  });
  const thresholdMs = auction.antiSnipeThreshold * 1000;
  const extensionMs = auction.antiSnipeExtension * 1000;
  const closingTime = new Date(lot.closesAt).getTime();
  const now = Date.now();
  if (closingTime - now <= thresholdMs) {
    const newClose = new Date(closingTime + extensionMs).toISOString();
    updateLot(lotId, { closesAt: newClose });
  }
  if (previousLeader && previousLeader !== teamId) {
    const prevBudget = getTeamBudget(previousLeader, auctionId);
    if (prevBudget) {
      updateTeamBudget(previousLeader, auctionId, { remainingBudget: prevBudget.remainingBudget + previousAmount });
    }
  }
  updateTeamBudget(teamId, auctionId, { remainingBudget: availableBudget - amount });
  return bid;
}

export function finalizeLot({ auctionId, lotId, actorId }) {
  const lot = listLotsByAuction(auctionId).find((l) => l.id === lotId);
  if (!lot) throw new Error('Lot not found');
  if (lot.status !== 'ACTIVE') throw new Error('Lot not active');
  const bids = listBidsByLot(lotId);
  if (!bids.length) {
    updateLot(lotId, { status: 'UNSOLD', closedAt: new Date().toISOString() });
    return { status: 'UNSOLD' };
  }
  const winningBid = bids.reduce((max, bid) => (bid.amount > max.amount ? bid : max), bids[0]);
  updateLot(lotId, {
    status: 'SOLD',
    closedAt: new Date().toISOString(),
    winnerTeamId: winningBid.teamId,
    soldPrice: winningBid.amount
  });
  createSale({
    id: createId(),
    lotId,
    auctionId,
    playerId: lot.playerId,
    teamId: winningBid.teamId,
    amount: winningBid.amount,
    soldAt: new Date().toISOString()
  });
  return { status: 'SOLD', winningBid };
}

export function pauseAuction(auctionId) {
  return updateAuction(auctionId, { status: 'PAUSED' });
}

export function resumeAuction(auctionId) {
  return updateAuction(auctionId, { status: 'ACTIVE' });
}

export function startAuction(auctionId) {
  const auction = updateAuction(auctionId, { status: 'ACTIVE', startedAt: new Date().toISOString() });
  if (!auction) throw new Error('Auction not found');
  return auction;
}

export function requestCompletion(auctionId, { override = false, actorId }) {
  const auction = ensureAuction(auctionId);
  const teams = listTeamsByAuction(auctionId);
  const rosterMap = new Map();
  for (const lot of listLotsByAuction(auctionId)) {
    if (lot.status === 'SOLD') {
      rosterMap.set(lot.winnerTeamId, (rosterMap.get(lot.winnerTeamId) || 0) + 1);
    }
  }
  const result = canCompleteAuction({ teams, rosterMap });
  if (!result.ok && !override) {
    throw new Error('Teams below roster minimum');
  }
  updateAuction(auctionId, { status: 'COMPLETED', completedAt: new Date().toISOString() });
  if (override) {
    saveAuditLog({ auctionId, actorId, action: 'OVERRIDE_COMPLETION', details: result.violations });
  }
  return { ok: true, violations: result.violations };
}

export function generateHighestSale(auctionId) {
  const sales = listSalesByAuction(auctionId);
  if (!sales.length) return null;
  const highest = sales.reduce((max, sale) => (sale.amount > max.amount ? sale : max), sales[0]);
  const player = findPlayerById(highest.playerId);
  return {
    player,
    sale: highest
  };
}

export function generateTeamSummary(auctionId) {
  const teams = listTeamsByAuction(auctionId);
  const budgets = teams.map((team) => {
    const budget = getTeamBudget(team.id, auctionId);
    const rosterLots = listLotsByAuction(auctionId).filter((lot) => lot.winnerTeamId === team.id);
    const totalSpent = rosterLots.reduce((sum, lot) => sum + (lot.soldPrice || 0), 0);
    const totalBase = rosterLots.reduce((sum, lot) => {
      const player = findPlayerById(lot.playerId);
      return sum + (player?.basePrice || 0);
    }, 0);
    return {
      team,
      totalSpent,
      totalBase,
      remainingBudget: budget?.remainingBudget ?? 0,
      players: rosterLots.map((lot) => {
        const player = findPlayerById(lot.playerId);
        return {
          player,
          soldPrice: lot.soldPrice,
          basePrice: player?.basePrice ?? 0
        };
      })
    };
  });
  return budgets;
}

export function exportTeamsSummaryCsv(auctionId) {
  const summary = generateTeamSummary(auctionId);
  const rows = [['Team', 'Player', 'Rating', 'Base Price', 'Sold Price']];
  for (const entry of summary) {
    for (const item of entry.players) {
      rows.push([
        entry.team.name,
        item.player?.name ?? 'Unknown',
        item.player?.rating ?? '',
        item.basePrice,
        item.soldPrice ?? ''
      ]);
    }
  }
  return rows.map((row) => row.join(',')).join('\n');
}
