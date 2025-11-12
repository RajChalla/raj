import { prisma } from '@/config/prisma';
import { computeBasePrice } from '@/utils/basePrice';
import { z } from 'zod';
import { getBasePriceTiers } from './settingsService';
import createHttpError from 'http-errors';

const playerSchema = z.object({
  id: z.string(),
  name: z.string(),
  position: z.string(),
  club: z.string(),
  nation: z.string(),
  rating: z.number().int(),
  rarity: z.string(),
  gender: z.string()
});

export type PlayerInput = z.infer<typeof playerSchema>;

function parseCsv(input: string): PlayerInput[] {
  const rows = input
    .trim()
    .split(/\r?\n/)
    .filter(Boolean);
  const [header, ...data] = rows;
  const headers = header.split(',').map((h) => h.trim().toLowerCase());
  const required = ['id', 'name', 'position', 'club', 'nation', 'rating', 'rarity', 'gender'];
  for (const key of required) {
    if (!headers.includes(key)) {
      throw new createHttpError.BadRequest('CSV missing required headers');
    }
  }
  return data.map((line) => {
    const values = line.split(',');
    const record: any = {};
    headers.forEach((h, idx) => {
      record[h] = values[idx]?.trim();
    });
    record.rating = Number(record.rating);
    return playerSchema.parse(record);
  });
}

export async function importPlayers(rawPlayers: PlayerInput[] | string, actorId: string) {
  let players: PlayerInput[];
  if (typeof rawPlayers === 'string') {
    players = parseCsv(rawPlayers);
  } else {
    players = rawPlayers.map((p) => playerSchema.parse(p));
  }
  const tiers = await getBasePriceTiers();
  const filtered = players.filter((player) => player.rarity === 'Gold Rare' && player.gender.toLowerCase() === 'men');
  if (filtered.length !== players.length) {
    throw new createHttpError.BadRequest('Import contains invalid rarity or gender');
  }
  await prisma.$transaction(async (tx) => {
    for (const player of filtered) {
      const basePrice = computeBasePrice(player.rating, tiers);
      await tx.player.upsert({
        where: { id: player.id },
        update: {
          name: player.name,
          position: player.position,
          club: player.club,
          nation: player.nation,
          rating: player.rating,
          rarity: player.rarity,
          gender: player.gender,
          basePrice
        },
        create: {
          id: player.id,
          name: player.name,
          position: player.position,
          club: player.club,
          nation: player.nation,
          rating: player.rating,
          rarity: player.rarity,
          gender: player.gender,
          basePrice
        }
      });
    }
    await tx.auditLog.create({
      data: {
        action: 'IMPORT_PLAYERS',
        detail: { count: filtered.length },
        userId: actorId
      }
    });
  });
}
