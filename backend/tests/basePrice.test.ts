import { computeBasePrice, defaultTiers } from '@/utils/basePrice';

describe('computeBasePrice', () => {
  it('returns highest tier', () => {
    expect(computeBasePrice(95)).toBe(300000);
  });

  it('returns proper tier', () => {
    expect(computeBasePrice(82)).toBe(70000);
    expect(computeBasePrice(79)).toBe(40000);
    expect(computeBasePrice(70)).toBe(10000);
  });

  it('supports custom tiers', () => {
    const tiers = [...defaultTiers];
    tiers[0] = { min: 90, max: 99, price: 999999 };
    expect(computeBasePrice(95, tiers)).toBe(999999);
  });
});
