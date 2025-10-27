import createHttpError from 'http-errors';
import { prisma } from '@/config/prisma';
import { AuctionStatus, LotStatus } from '@prisma/client';
import { AuctionSocketGateway } from '@/ws/auctionGateway';
import { validateBid } from '@/utils/bidRules';

export class BidService {
  constructor(private gateway: AuctionSocketGateway) {}

  async placeBid(lotId: string, teamId: string, amount: number) {
    const lot = await prisma.lot.findUnique({
      where: { id: lotId },
      include: {
        auction: true,
        player: true,
        currentBid: true
      }
    });
    if (!lot) {
      throw new createHttpError.NotFound('Lot not found');
    }
    if (lot.status !== LotStatus.ACTIVE) {
      throw new createHttpError.BadRequest('Lot is not active');
    }
    if (lot.endsAt && lot.endsAt.getTime() <= Date.now()) {
      throw new createHttpError.BadRequest('Lot already ended');
    }
    const auction = lot.auction;
    if (auction.status !== AuctionStatus.RUNNING) {
      throw new createHttpError.BadRequest('Auction not running');
    }
    const team = await prisma.team.findUnique({ where: { id: teamId } });
    if (!team || team.auctionId !== lot.auctionId) {
      throw new createHttpError.BadRequest('Team not in auction');
    }
    try {
      validateBid(amount, { lot, auction, team, currentBid: lot.currentBid });
    } catch (err: any) {
      throw new createHttpError.BadRequest(err.message);
    }
    const bid = await prisma.$transaction(async (tx) => {
      const created = await tx.bid.create({
        data: {
          lotId,
          teamId,
          amount
        }
      });
      await tx.lot.update({
        where: { id: lotId },
        data: { currentBidId: created.id }
      });
      const remainingBudget = team.remainingBudget - amount;
      this.gateway.broadcastBid(lot.auctionId, {
        bidId: created.id,
        lotId,
        teamId,
        amount,
        remainingBudget
      });
      if (lot.endsAt) {
        const msRemaining = lot.endsAt.getTime() - Date.now();
        if (msRemaining <= auction.antiSnipeWindow * 1000) {
          const newEnd = new Date(Date.now() + auction.antiSnipeExtend * 1000);
          await tx.lot.update({ where: { id: lotId }, data: { endsAt: newEnd } });
          this.gateway.broadcastTick(lot.auctionId, lotId, newEnd.getTime());
        }
      }
      return created;
    });
    return bid;
  }
}
