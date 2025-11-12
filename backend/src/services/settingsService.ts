import { prisma } from '@/config/prisma';
import { RatingTier, defaultTiers } from '@/utils/basePrice';

const BASE_PRICE_KEY = 'basePriceTiers';

export async function getBasePriceTiers(): Promise<RatingTier[]> {
  const record = await prisma.setting.findUnique({ where: { key: BASE_PRICE_KEY } });
  if (!record) {
    return defaultTiers;
  }
  return (record.value as RatingTier[]) ?? defaultTiers;
}

export async function updateBasePriceTiers(tiers: RatingTier[], actorId: string) {
  await prisma.$transaction(async (tx) => {
    await tx.setting.upsert({
      where: { key: BASE_PRICE_KEY },
      update: { value: tiers },
      create: { key: BASE_PRICE_KEY, value: tiers }
    });
    await tx.auditLog.create({
      data: {
        action: 'SET_BASE_PRICE_TIERS',
        detail: tiers,
        userId: actorId
      }
    });
  });
}
