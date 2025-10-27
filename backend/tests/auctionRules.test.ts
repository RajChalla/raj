import { teamsBelowMinimum } from '@/utils/auctionRules';

describe('teamsBelowMinimum', () => {
  it('identifies teams under 19', () => {
    const teams = [
      { id: 'a', rosterCount: 18 },
      { id: 'b', rosterCount: 19 },
      { id: 'c', rosterCount: 25 }
    ];
    const result = teamsBelowMinimum(teams);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('a');
  });
});
