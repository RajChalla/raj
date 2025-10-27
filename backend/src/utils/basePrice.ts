export type RatingTier = {
  min: number;
  max: number;
  price: number;
};

export const defaultTiers: RatingTier[] = [
  { min: 90, max: 99, price: 300000 },
  { min: 87, max: 89, price: 200000 },
  { min: 84, max: 86, price: 120000 },
  { min: 81, max: 83, price: 70000 },
  { min: 78, max: 80, price: 40000 },
  { min: 75, max: 77, price: 20000 },
  { min: 0, max: 74, price: 10000 }
];

export function computeBasePrice(rating: number, tiers: RatingTier[] = defaultTiers): number {
  const match = tiers.find((tier) => rating >= tier.min && rating <= tier.max);
  return match ? match.price : defaultTiers[defaultTiers.length - 1].price;
}
