import createHttpError from 'http-errors';
import { AuctionStatus, LotStatus, Prisma } from '@prisma/client';
import { prisma } from '@/config/prisma';
import { getBasePriceTiers } from './settingsService';
import { AuctionSocketGateway } from '@/ws/auctionGateway';
import { teamsBelowMinimum } from '@/utils/auctionRules';

export class AuctionService {
  constructor(private gateway: AuctionSocketGateway) {}

  async createAuction(name: string, actorId: string) {
    const tiers = await getBasePriceTiers();
    const auction = await prisma.auction.create({
      data: {
        name,
        settingsSnapshot: { tiers }
      }
    });
    await prisma.auditLog.create({
      data: {
        action: 'CREATE_AUCTION',
        detail: { auctionId: auction.id, name },
        userId: actorId,
        auctionId: auction.id
      }
    });
    return auction;
  }

  async setTeams(auctionId: string, teams: { name: string; ownerId?: string; budget: number }[], actorId: string) {
    return prisma.$transaction(async (tx) => {
      await tx.team.deleteMany({ where: { auctionId } });
      const created = await Promise.all(
        teams.map((team) =>
          tx.team.create({
            data: {
              auctionId,
              name: team.name,
              ownerId: team.ownerId,
              initialBudget: team.budget,
              remainingBudget: team.budget
            }
          })
        )
      );
      await tx.auditLog.create({
        data: {
          action: 'SET_TEAMS',
          detail: created.map((t) => ({ id: t.id, name: t.name, budget: t.initialBudget })),
          userId: actorId,
          auctionId
        }
      });
      return created;
    });
  }

  async queueLots(auctionId: string, playerIds: string[], actorId: string) {
    const players = await prisma.player.findMany({ where: { id: { in: playerIds } } });
    if (players.length !== playerIds.length) {
      throw new createHttpError.BadRequest('Unknown players in queue');
    }
    const lots = await prisma.$transaction(async (tx) => {
      return Promise.all(
        players.map((player, index) =>
          tx.lot.create({
            data: {
              auctionId,
              playerId: player.id,
              status: LotStatus.QUEUED,
              reservePrice: player.basePrice,
              sale: {
                create: {
                  auctionId,
                  playerId: player.id,
                  basePrice: player.basePrice
                }
              },
              createdAt: new Date(Date.now() + index)
            }
          })
        )
      );
    });
    await prisma.auditLog.create({
      data: {
        action: 'QUEUE_LOTS',
        detail: { auctionId, count: lots.length },
        userId: actorId,
        auctionId
      }
    });
    return lots;
  }

  async startAuction(auctionId: string, actorId: string) {
    const auction = await prisma.auction.update({
      where: { id: auctionId },
      data: {
        status: AuctionStatus.RUNNING,
        startTime: new Date()
      }
    });
    await prisma.auditLog.create({
      data: {
        action: 'START_AUCTION',
        detail: { auctionId },
        userId: actorId,
        auctionId
      }
    });
    this.gateway.broadcastAuctionStatus(auctionId, AuctionStatus.RUNNING);
    return auction;
  }

  async pauseAuction(auctionId: string, actorId: string) {
    const auction = await prisma.auction.update({
      where: { id: auctionId },
      data: {
        status: AuctionStatus.PAUSED
      }
    });
    await prisma.auditLog.create({
      data: {
        action: 'PAUSE_AUCTION',
        detail: { auctionId },
        userId: actorId,
        auctionId
      }
    });
    this.gateway.broadcastAuctionStatus(auctionId, AuctionStatus.PAUSED);
    return auction;
  }

  async resumeAuction(auctionId: string, actorId: string) {
    const auction = await prisma.auction.update({
      where: { id: auctionId },
      data: {
        status: AuctionStatus.RUNNING
      }
    });
    await prisma.auditLog.create({
      data: {
        action: 'RESUME_AUCTION',
        detail: { auctionId },
        userId: actorId,
        auctionId
      }
    });
    this.gateway.broadcastAuctionStatus(auctionId, AuctionStatus.RUNNING);
    return auction;
  }

  async completeAuction(auctionId: string, actorId: string, override: boolean, reason?: string) {
    const teams = await prisma.team.findMany({ where: { auctionId } });
    const short = teamsBelowMinimum(teams);
    if (short.length > 0 && !override) {
      throw new createHttpError.BadRequest('Cannot complete auction. Teams under roster minimum.');
    }
    if (short.length > 0 && override) {
      await prisma.auditLog.create({
        data: {
          action: 'COMPLETE_AUCTION_OVERRIDE',
          detail: { auctionId, teams: short.map((t) => ({ id: t.id, rosterCount: t.rosterCount })), reason },
          userId: actorId,
          auctionId
        }
      });
    }
    const auction = await prisma.auction.update({
      where: { id: auctionId },
      data: { status: AuctionStatus.COMPLETED, endTime: new Date() }
    });
    this.gateway.broadcastAuctionStatus(auctionId, AuctionStatus.COMPLETED);
    return auction;
  }

  async activateNextLot(auctionId: string, durationSeconds: number) {
    const existingActive = await prisma.lot.findFirst({ where: { auctionId, status: LotStatus.ACTIVE } });
    if (existingActive) {
      return existingActive;
    }
    const nextLot = await prisma.lot.findFirst({
      where: { auctionId, status: LotStatus.QUEUED },
      orderBy: { createdAt: 'asc' },
      include: { player: true }
    });
    if (!nextLot) {
      return null;
    }
    const startsAt = new Date();
    const endsAt = new Date(startsAt.getTime() + durationSeconds * 1000);
    const updated = await prisma.lot.update({
      where: { id: nextLot.id },
      data: { status: LotStatus.ACTIVE, startsAt, endsAt }
    });
    this.gateway.broadcastLotActivated(auctionId, { ...updated, player: nextLot.player });
    this.gateway.startTickingLot(updated);
    return updated;
  }

  async getActiveLot(auctionId: string) {
    return prisma.lot.findFirst({
      where: { auctionId, status: LotStatus.ACTIVE },
      include: { player: true, currentBid: true }
    });
  }

  async markLotUnsold(lotId: string, actorId: string) {
    const lot = await prisma.lot.update({
      where: { id: lotId },
      data: { status: LotStatus.UNSOLD, endsAt: new Date() },
      include: { auction: true, player: true, sale: true }
    });
    if (!lot.sale) {
      await prisma.sale.upsert({
        where: { lotId },
        create: {
          lotId,
          auctionId: lot.auctionId,
          playerId: lot.playerId,
          basePrice: lot.reservePrice
        },
        update: {}
      });
    }
    await prisma.auditLog.create({
      data: {
        action: 'MARK_UNSOLD',
        detail: { lotId },
        userId: actorId,
        auctionId: lot.auctionId
      }
    });
    this.gateway.broadcastLotSold(lot.auctionId, {
      lotId: lot.id,
      status: LotStatus.UNSOLD,
      soldPrice: null,
      teamId: null
    });
    return lot;
  }

  async settleSale(lotId: string, winningBid: Prisma.BidUncheckedCreateInput) {
    const sale = await prisma.sale.update({
      where: { lotId },
      data: {
        teamId: winningBid.teamId,
        soldPrice: winningBid.amount
      }
    });
    await prisma.team.update({
      where: { id: winningBid.teamId },
      data: {
        rosterCount: { increment: 1 },
        remainingBudget: { decrement: winningBid.amount }
      }
    });
    return sale;
  }

  async closeLotAsSold(lotId: string, actorId: string) {
    const lot = await prisma.lot.findUnique({
      where: { id: lotId },
      include: { currentBid: true, auction: true, sale: true }
    });
    if (!lot) {
      throw new createHttpError.NotFound('Lot not found');
    }
    if (!lot.currentBid) {
      throw new createHttpError.BadRequest('No winning bid to settle');
    }
    await prisma.$transaction(async (tx) => {
      await tx.lot.update({
        where: { id: lotId },
        data: { status: LotStatus.SOLD, endsAt: new Date() }
      });
      await tx.sale.update({
        where: { lotId },
        data: {
          teamId: lot.currentBid!.teamId,
          soldPrice: lot.currentBid!.amount
        }
      });
      await tx.team.update({
        where: { id: lot.currentBid!.teamId },
        data: {
          rosterCount: { increment: 1 },
          remainingBudget: { decrement: lot.currentBid!.amount }
        }
      });
    });
    await prisma.auditLog.create({
      data: {
        action: 'MARK_SOLD',
        detail: { lotId, bidId: lot.currentBid.id },
        userId: actorId,
        auctionId: lot.auctionId
      }
    });
    this.gateway.broadcastLotSold(lot.auctionId, {
      lotId: lot.id,
      status: LotStatus.SOLD,
      soldPrice: lot.currentBid.amount,
      teamId: lot.currentBid.teamId
    });
  }
}
