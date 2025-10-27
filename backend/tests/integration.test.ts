import request from 'supertest';
import { newDb } from 'pg-mem';
import fs from 'fs';
import path from 'path';
import { Role } from '@prisma/client';
import { hashPassword } from '@/utils/auth';

const db = newDb({ autoCreateForeignKeyIndices: true });
const pg = db.adapters.createPg();

db.public.registerFunction({ name: 'current_database', returns: 'text', implementation: () => 'test' });

jest.mock('pg', () => pg);

const { prisma } = require('@/config/prisma');
const { createApp } = require('@/app');
const { AuctionSocketGateway } = require('@/ws/auctionGateway');

describe('Auction integration', () => {
  let agentAdmin: request.SuperTest<request.Test>;
  let agentOwner1: request.SuperTest<request.Test>;
  let agentOwner2: request.SuperTest<request.Test>;
  let auctionId: string;
  let lotId: string;

  beforeAll(async () => {
    const sql = fs.readFileSync(path.join(__dirname, '../prisma/migrations/0001_init/migration.sql'), 'utf-8');
    db.public.none(sql);
    await prisma.$connect();

    const [adminHash, owner1Hash, owner2Hash] = await Promise.all([
      hashPassword('adminpass'),
      hashPassword('owner1pass'),
      hashPassword('owner2pass')
    ]);

    await prisma.user.createMany({
      data: [
        { id: 'admin1', email: 'admin@example.com', passwordHash: adminHash, role: Role.ADMIN, displayName: 'Admin' },
        { id: 'owner1', email: 'owner1@example.com', passwordHash: owner1Hash, role: Role.OWNER, displayName: 'Owner One' },
        { id: 'owner2', email: 'owner2@example.com', passwordHash: owner2Hash, role: Role.OWNER, displayName: 'Owner Two' }
      ]
    });

    const app = createApp(new AuctionSocketGateway());
    agentAdmin = request.agent(app);
    agentOwner1 = request.agent(app);
    agentOwner2 = request.agent(app);
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  it('runs complete flow', async () => {
    await agentAdmin.post('/api/auth/login').send({ email: 'admin@example.com', password: 'adminpass' }).expect(200);
    await agentOwner1.post('/api/auth/login').send({ email: 'owner1@example.com', password: 'owner1pass' }).expect(200);
    await agentOwner2.post('/api/auth/login').send({ email: 'owner2@example.com', password: 'owner2pass' }).expect(200);

    const auctionRes = await agentAdmin.post('/api/auctions').send({ name: 'Test Auction' }).expect(201);
    auctionId = auctionRes.body.id;

    await agentAdmin
      .post(`/api/auctions/${auctionId}/teams`)
      .send({ teams: [
        { name: 'Team One', ownerId: 'owner1', budget: 400000 },
        { name: 'Team Two', ownerId: 'owner2', budget: 400000 }
      ] })
      .expect(200);

    await agentAdmin
      .post('/api/players/import')
      .send({
        players: [
          { id: 'p1', name: 'Player One', position: 'ST', club: 'Club', nation: 'Nation', rating: 90, rarity: 'Gold Rare', gender: 'Men' },
          { id: 'p2', name: 'Player Two', position: 'GK', club: 'Club', nation: 'Nation', rating: 88, rarity: 'Gold Rare', gender: 'Men' }
        ]
      })
      .expect(204);

    const lotsRes = await agentAdmin.post(`/api/auctions/${auctionId}/lots`).send({ playerIds: ['p1', 'p2'] }).expect(200);
    lotId = lotsRes.body[0].id;

    await agentAdmin.patch(`/api/auctions/${auctionId}/start`).expect(200);
    await agentAdmin.post(`/api/auctions/${auctionId}/lots/activate`).send({ durationSeconds: 60 }).expect(200);

    await agentOwner1.post(`/api/lots/${lotId}/bids`).send({ amount: 320000 }).expect(400);
    await agentOwner1.post(`/api/lots/${lotId}/bids`).send({ amount: 300000 }).expect(201);
    await agentOwner2.post(`/api/lots/${lotId}/bids`).send({ amount: 301000 }).expect(400);
    await agentOwner2.post(`/api/lots/${lotId}/bids`).send({ amount: 310000 }).expect(201);

    await agentAdmin.patch(`/api/lots/${lotId}/sold`).expect(204);

    const highest = await agentAdmin.get(`/api/auctions/${auctionId}/reports/highest-sale`).expect(200);
    expect(highest.body.soldPrice).toBe(310000);
    expect(highest.body.teamId).toBeDefined();

    const summary = await agentAdmin.get(`/api/auctions/${auctionId}/reports/teams-summary`).expect(200);
    expect(summary.body).toHaveLength(2);
    const teamTwo = summary.body.find((team: any) => team.teamName === 'Team Two');
    expect(teamTwo.players[0].soldPrice).toBe(310000);
    expect(teamTwo.remainingBudget).toBe(400000 - 310000);
  });
});
