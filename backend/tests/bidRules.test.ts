import { validateBid } from '@/utils/bidRules';
import { Auction, Bid, Lot, Team } from '@prisma/client';

describe('validateBid', () => {
  const auction = { bidIncrement: 1000 } as Auction;
  const lot = { reservePrice: 5000 } as Lot;
  const team = { remainingBudget: 6000 } as Team;

  it('rejects below reserve', () => {
    expect(() => validateBid(4000, { auction, lot, team, currentBid: null })).toThrow('Bid below reserve price');
  });

  it('rejects over budget', () => {
    expect(() => validateBid(7000, { auction, lot, team, currentBid: null })).toThrow('Bid exceeds remaining budget');
  });

  it('rejects non-increment', () => {
    const currentBid = { amount: 5500 } as Bid;
    expect(() => validateBid(6000, { auction, lot, team: { remainingBudget: 10000 } as Team, currentBid })).toThrow('Bid must respect minimum increment');
  });

  it('passes valid bid', () => {
    expect(() => validateBid(6500, { auction, lot, team: { remainingBudget: 10000 } as Team, currentBid: { amount: 5500 } as Bid })).not.toThrow();
  });
});
