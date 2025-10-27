#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-var-requires */
const { PrismaClient, Role } = require('@prisma/client');
const bcrypt = require('bcryptjs');

const prisma = new PrismaClient();

const players = [
  {
    id: 'fc26-001',
    name: 'Kylian Mbappé',
    position: 'ST',
    club: 'Paris SG',
    nation: 'France',
    rating: 91,
    rarity: 'Gold Rare',
    gender: 'Men',
    basePrice: 300000
  },
  {
    id: 'fc26-002',
    name: 'Kevin De Bruyne',
    position: 'CM',
    club: 'Manchester City',
    nation: 'Belgium',
    rating: 91,
    rarity: 'Gold Rare',
    gender: 'Men',
    basePrice: 300000
  },
  {
    id: 'fc26-003',
    name: 'Virgil van Dijk',
    position: 'CB',
    club: 'Liverpool',
    nation: 'Netherlands',
    rating: 90,
    rarity: 'Gold Rare',
    gender: 'Men',
    basePrice: 300000
  }
];

async function main() {
  console.log('Seeding FC26 data...');
  await prisma.user.upsert({
    where: { email: 'admin@example.com' },
    update: {},
    create: {
      email: 'admin@example.com',
      displayName: 'Admin',
      passwordHash: await bcrypt.hash('adminpass', 10),
      role: Role.ADMIN
    }
  });
  await prisma.user.upsert({
    where: { email: 'owner1@example.com' },
    update: {},
    create: {
      email: 'owner1@example.com',
      displayName: 'Owner One',
      passwordHash: await bcrypt.hash('owner1pass', 10),
      role: Role.OWNER
    }
  });
  await prisma.user.upsert({
    where: { email: 'owner2@example.com' },
    update: {},
    create: {
      email: 'owner2@example.com',
      displayName: 'Owner Two',
      passwordHash: await bcrypt.hash('owner2pass', 10),
      role: Role.OWNER
    }
  });

  for (const player of players) {
    await prisma.player.upsert({
      where: { id: player.id },
      update: player,
      create: player
    });
  }

  console.log('Seed complete');
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
