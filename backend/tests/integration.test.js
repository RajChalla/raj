import { test, before, after } from 'node:test';
import assert from 'node:assert';

process.env.NODE_ENV = 'test';

const { server, tickTimer } = await import('../src/server.js');
const { resetAll } = await import('../src/services/storeService.js');
const { ensureSeedUsers } = await import('../src/services/authService.js');

let baseUrl;

async function request(path, { method = 'GET', token, body } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });
  const data = await response.json();
  return { status: response.status, data };
}

before(async () => {
  resetAll();
  ensureSeedUsers();
  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  baseUrl = `http://127.0.0.1:${port}`;
});

after(() => {
  server.close();
  clearInterval(tickTimer);
});

test('full auction flow with two owners', async () => {
  const adminLogin = await request('/api/auth/login', {
    method: 'POST',
    body: { username: 'admin', password: 'admin123' }
  });
  assert.equal(adminLogin.status, 200);
  const adminToken = adminLogin.data.token;

  const owner1Login = await request('/api/auth/login', {
    method: 'POST',
    body: { username: 'owner1', password: 'owner123' }
  });
  const owner2Login = await request('/api/auth/login', {
    method: 'POST',
    body: { username: 'owner2', password: 'owner123' }
  });
  const owner1Token = owner1Login.data.token;
  const owner2Token = owner2Login.data.token;

  const importResp = await request('/api/players/import', {
    method: 'POST',
    token: adminToken,
    body: {
      records: [
        {
          name: 'Erling Haaland',
          position: 'ST',
          club: 'Manchester City',
          nation: 'Norway',
          rating: '92',
          rarity: 'Gold Rare',
          gender: 'men'
        },
        {
          name: 'Kylian Mbappé',
          position: 'ST',
          club: 'Paris SG',
          nation: 'France',
          rating: '91',
          rarity: 'Gold Rare',
          gender: 'men'
        }
      ]
    }
  });
  assert.equal(importResp.status, 200);

  const playersResp = await request('/api/players', { token: adminToken });
  assert.equal(playersResp.status, 200);
  const players = playersResp.data.players;
  assert.ok(players.length >= 2);

  const auctionResp = await request('/api/auctions', {
    method: 'POST',
    token: adminToken,
    body: { name: 'Test Auction', increment: 1000, antiSnipeThreshold: 2, antiSnipeExtension: 4 }
  });
  assert.equal(auctionResp.status, 201);
  const auctionId = auctionResp.data.auction.id;

  const teamResp = await request(`/api/auctions/${auctionId}/teams`, {
    method: 'POST',
    token: adminToken,
    body: {
      teams: [
        { name: 'Blue', ownerUserId: owner1Login.data.user.id, budget: 1000000 },
        { name: 'Red', ownerUserId: owner2Login.data.user.id, budget: 1000000 }
      ]
    }
  });
  assert.equal(teamResp.status, 200);
  const [teamBlue, teamRed] = teamResp.data.teams;

  const queueResp = await request(`/api/auctions/${auctionId}/lots/queue`, {
    method: 'POST',
    token: adminToken,
    body: { playerIds: players.slice(0, 2).map((p) => p.id) }
  });
  assert.equal(queueResp.status, 200);

  const startResp = await request(`/api/auctions/${auctionId}/start`, {
    method: 'PATCH',
    token: adminToken
  });
  assert.equal(startResp.status, 200);

  const nextResp = await request(`/api/auctions/${auctionId}/lots/next`, {
    method: 'PATCH',
    token: adminToken
  });
  assert.equal(nextResp.status, 200);
  const lot = nextResp.data.lot;
  assert.ok(lot);

  const firstBid = await request(`/api/lots/${lot.id}/bids`, {
    method: 'POST',
    token: owner1Token,
    body: { amount: lot.reserve, auctionId, teamId: teamBlue.id }
  });
  assert.equal(firstBid.status, 201);

  const secondBid = await request(`/api/lots/${lot.id}/bids`, {
    method: 'POST',
    token: owner2Token,
    body: { amount: lot.reserve + 1000, auctionId, teamId: teamRed.id }
  });
  assert.equal(secondBid.status, 201);

  const finalize = await request(`/api/lots/${lot.id}/finalize`, {
    method: 'PATCH',
    token: adminToken,
    body: { auctionId }
  });
  assert.equal(finalize.status, 200);
  assert.equal(finalize.data.status, 'SOLD');

  const failComplete = await request(`/api/auctions/${auctionId}/complete`, {
    method: 'PATCH',
    token: adminToken,
    body: { override: false }
  });
  assert.equal(failComplete.status, 400);

  const complete = await request(`/api/auctions/${auctionId}/complete`, {
    method: 'PATCH',
    token: adminToken,
    body: { override: true }
  });
  assert.equal(complete.status, 200);
  assert.ok(Array.isArray(complete.data.violations));

  const highestSale = await request(`/api/auctions/${auctionId}/reports/highest-sale`, {
    token: adminToken
  });
  assert.equal(highestSale.status, 200);
  assert.ok(highestSale.data.report);

  const summary = await request(`/api/auctions/${auctionId}/reports/teams-summary`, {
    token: adminToken
  });
  assert.equal(summary.status, 200);
  assert.equal(summary.data.summary.length, 2);

  const csv = await request(`/api/auctions/${auctionId}/export.csv`, {
    token: adminToken
  });
  assert.equal(csv.status, 200);
  assert.ok(csv.data.csv.includes('Sold Price'));
});
