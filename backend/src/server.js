import http from 'http';
import { Router } from './http/router.js';
import { readBody } from './http/body.js';
import { sendJson, unauthorized, forbidden, badRequest } from './http/response.js';
import { ensureSeedUsers, login, authenticate } from './services/authService.js';
import { importPlayers, getPlayers } from './services/playerService.js';
import {
  createAuctionWithSettings,
  setTeamsAndBudgets,
  queueLots,
  activateNextLot,
  placeBid,
  finalizeLot,
  startAuction,
  pauseAuction,
  resumeAuction,
  requestCompletion,
  generateHighestSale,
  generateTeamSummary,
  exportTeamsSummaryCsv
} from './services/auctionService.js';
import { getBasePriceTiers, setBasePriceTiers, listTeamsByAuction, getActiveLot, listBidsByLot } from './services/storeService.js';
import { AuctionSocketServer } from './ws/server.js';
import { store } from './data/store.js';
import { parse } from 'url';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const router = new Router();
const socketServer = new AuctionSocketServer();

ensureSeedUsers();

function getAuthToken(req) {
  const header = req.headers['authorization'];
  if (!header) return null;
  const [scheme, token] = header.split(' ');
  if (scheme !== 'Bearer') return null;
  return token;
}

function requireUser(req, res) {
  const token = getAuthToken(req);
  const user = authenticate(token);
  if (!user) {
    unauthorized(res);
    return null;
  }
  return user;
}

function requireRole(req, res, roles) {
  const user = requireUser(req, res);
  if (!user) return null;
  if (!roles.includes(user.role)) {
    forbidden(res);
    return null;
  }
  return user;
}

router.register('POST', 'api/auth/login', async (req, res) => {
  const body = await readBody(req);
  if (!body) return badRequest(res, 'Invalid body');
  try {
    const result = login(body);
    sendJson(res, 200, result);
  } catch (err) {
    unauthorized(res);
  }
});

router.register('GET', 'api/auth/me', (req, res) => {
  const user = requireUser(req, res);
  if (!user) return;
  sendJson(res, 200, { user });
});

router.register('POST', 'api/auth/logout', (_req, res) => {
  sendJson(res, 200, { ok: true });
});

router.register('POST', 'api/players/import', async (req, res) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!body) return badRequest(res, 'Payload required');
  try {
    const players = importPlayers(body.records ?? body);
    sendJson(res, 200, { imported: players.length });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('GET', 'api/players', (req, res) => {
  const user = requireUser(req, res);
  if (!user) return;
  sendJson(res, 200, { players: getPlayers() });
});

router.register('POST', 'api/auctions', async (req, res) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!body || !body.name) return badRequest(res, 'Name required');
  const auction = createAuctionWithSettings(body);
  sendJson(res, 201, { auction });
});

router.register('POST', 'api/auctions/:id/teams', async (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!body || !Array.isArray(body.teams)) return badRequest(res, 'Teams array required');
  try {
    const teams = setTeamsAndBudgets(params.id, body.teams, user.id);
    sendJson(res, 200, { teams });
    socketServer.broadcast(params.id, 'TEAM_UPDATED', { teams });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('GET', 'api/auctions/:id/teams', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN', 'OWNER']);
  if (!user) return;
  const teams = listTeamsByAuction(params.id);
  sendJson(res, 200, { teams });
});

router.register('GET', 'api/auctions/:id/lots/active', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN', 'OWNER']);
  if (!user) return;
  const lot = getActiveLot(params.id);
  const bids = lot ? listBidsByLot(lot.id) : [];
  sendJson(res, 200, { lot, bids });
});

router.register('PATCH', 'api/settings/basePriceTiers', async (req, res) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!Array.isArray(body?.tiers)) return badRequest(res, 'Tiers array required');
  setBasePriceTiers(body.tiers);
  sendJson(res, 200, { tiers: getBasePriceTiers() });
});

router.register('PATCH', 'api/auctions/:id/start', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  try {
    const auction = startAuction(params.id);
    sendJson(res, 200, { auction });
    socketServer.broadcast(params.id, 'AUCTION_STATUS_CHANGED', { status: 'ACTIVE' });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('PATCH', 'api/auctions/:id/pause', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const auction = pauseAuction(params.id);
  sendJson(res, 200, { auction });
  socketServer.broadcast(params.id, 'AUCTION_STATUS_CHANGED', { status: 'PAUSED' });
});

router.register('PATCH', 'api/auctions/:id/resume', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const auction = resumeAuction(params.id);
  sendJson(res, 200, { auction });
  socketServer.broadcast(params.id, 'AUCTION_STATUS_CHANGED', { status: 'ACTIVE' });
});

router.register('PATCH', 'api/auctions/:id/complete', async (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  try {
    const result = requestCompletion(params.id, { override: Boolean(body?.override), actorId: user.id });
    sendJson(res, 200, result);
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('POST', 'api/auctions/:id/lots/queue', async (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!Array.isArray(body?.playerIds)) return badRequest(res, 'playerIds required');
  try {
    queueLots(params.id, body.playerIds);
    sendJson(res, 200, { queued: body.playerIds.length });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('PATCH', 'api/auctions/:id/lots/next', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  try {
    const lot = activateNextLot(params.id);
    if (!lot) return sendJson(res, 200, { lot: null });
    sendJson(res, 200, { lot });
    socketServer.broadcast(params.id, 'LOT_ACTIVATED', { lot });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('POST', 'api/lots/:lotId/bids', async (req, res, { params }) => {
  const user = requireRole(req, res, ['OWNER', 'ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  if (!body || typeof body.amount !== 'number' || !body.auctionId || !body.teamId) {
    return badRequest(res, 'amount, auctionId, teamId required');
  }
  try {
    const bid = placeBid({
      auctionId: body.auctionId,
      lotId: params.lotId,
      teamId: body.teamId,
      amount: body.amount,
      actorId: user.id
    });
    sendJson(res, 201, { bid });
    socketServer.broadcast(body.auctionId, 'BID_PLACED', { bid });
    socketServer.broadcast(body.auctionId, 'TEAM_UPDATED', { teamId: body.teamId });
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('PATCH', 'api/lots/:lotId/finalize', async (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const body = await readBody(req);
  try {
    const result = finalizeLot({ auctionId: body.auctionId, lotId: params.lotId, actorId: user.id });
    sendJson(res, 200, result);
    socketServer.broadcast(body.auctionId, 'LOT_SOLD', result);
    socketServer.broadcast(body.auctionId, 'TEAM_UPDATED', result);
  } catch (err) {
    badRequest(res, err.message);
  }
});

router.register('GET', 'api/auctions/:id/reports/highest-sale', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN', 'OWNER']);
  if (!user) return;
  const report = generateHighestSale(params.id);
  sendJson(res, 200, { report });
});

router.register('GET', 'api/auctions/:id/reports/teams-summary', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN', 'OWNER']);
  if (!user) return;
  const summary = generateTeamSummary(params.id);
  sendJson(res, 200, { summary });
});

router.register('GET', 'api/auctions/:id/export.csv', (req, res, { params }) => {
  const user = requireRole(req, res, ['ADMIN']);
  if (!user) return;
  const csv = exportTeamsSummaryCsv(params.id);
  sendJson(res, 200, { csv });
});

router.register('GET', '', async (req, res) => {
  const { pathname } = parse(req.url);
  const isHead = req.method === 'HEAD';
  if (pathname.startsWith('/app.js')) {
    const js = isHead
      ? null
      : await readFile(join(__dirname, '../../web/app.js'), 'utf8');
    res.writeHead(200, { 'Content-Type': 'application/javascript' });
    res.end(isHead ? undefined : js);
    return;
  }
  if (pathname.startsWith('/styles.css')) {
    const css = isHead
      ? null
      : await readFile(join(__dirname, '../../web/styles.css'), 'utf8');
    res.writeHead(200, { 'Content-Type': 'text/css' });
    res.end(isHead ? undefined : css);
    return;
  }
  if (pathname === '/' || pathname.startsWith('/app')) {
    const html = isHead
      ? null
      : await readFile(join(__dirname, '../../web/index.html'), 'utf8');
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(isHead ? undefined : html);
    return;
  }
  sendJson(res, 404, { error: 'Not Found' });
});

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Allow-Methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS'
    });
    res.end();
    return;
  }
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS');
  router.handle(req, res, {});
});

socketServer.attach(server);

const tickTimer = setInterval(() => {
  const auctions = store.snapshot.auctions;
  for (const auction of auctions) {
    const lot = getActiveLot(auction.id);
    if (lot) {
      socketServer.broadcast(auction.id, 'TICK', {
        lotId: lot.id,
        closesAt: lot.closesAt
      });
    }
  }
}, 1000);

const PORT = process.env.PORT || 4000;

if (process.env.NODE_ENV !== 'test') {
  server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

export { server, tickTimer };
