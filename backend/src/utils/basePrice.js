import { getBasePriceTiers } from '../services/storeService.js';

export function computeBasePrice(rating, tiers = getBasePriceTiers()) {
  const sorted = [...tiers].sort((a, b) => b.min - a.min);
  for (const tier of sorted) {
    if (rating >= tier.min && rating <= tier.max) {
      return tier.price;
    }
  }
  return sorted[sorted.length - 1]?.price ?? 10000;
}
