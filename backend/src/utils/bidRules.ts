import { Auction, Bid, Lot, Team } from '@prisma/client';

export type BidContext = {
  lot: Lot & { reservePrice: number; endsAt: Date | null };
  auction: Auction;
  team: Team;
  currentBid?: Bid | null;
};

export function validateBid(amount: number, context: BidContext) {
  if (amount < context.lot.reservePrice) {
    throw new Error('Bid below reserve price');
  }
  if (context.team.remainingBudget < amount) {
    throw new Error('Bid exceeds remaining budget');
  }
  const currentHigh = context.currentBid?.amount ?? 0;
  if (amount <= currentHigh) {
    throw new Error('Bid must exceed current high bid');
  }
  if (currentHigh > 0 && amount - currentHigh < context.auction.bidIncrement) {
    throw new Error('Bid must respect minimum increment');
  }
}
