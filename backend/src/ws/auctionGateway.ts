import { Server } from 'socket.io';
import { Lot, LotStatus } from '@prisma/client';
import { env } from '@/config/env';

export class AuctionSocketGateway {
  private io?: Server;
  private timers: Map<string, NodeJS.Timeout> = new Map();

  setServer(io: Server) {
    this.io = io;
  }

  broadcastAuctionStatus(auctionId: string, status: string) {
    this.io?.to(`auction:${auctionId}`).emit('AUCTION_STATUS_CHANGED', { auctionId, status });
  }

  broadcastLotActivated(auctionId: string, lot: Lot & { player?: any }) {
    this.io?.to(`auction:${auctionId}`).emit('LOT_ACTIVATED', lot);
  }

  broadcastBid(auctionId: string, payload: { bidId: string; lotId: string; teamId: string; amount: number; remainingBudget: number }) {
    this.io?.to(`auction:${auctionId}`).emit('BID_PLACED', payload);
    this.io?.to(`auction:${auctionId}`).emit('TEAM_UPDATED', {
      teamId: payload.teamId,
      remainingBudget: payload.remainingBudget
    });
  }

  broadcastLotSold(auctionId: string, payload: { lotId: string; status: LotStatus; soldPrice: number | null; teamId: string | null }) {
    this.io?.to(`auction:${auctionId}`).emit('LOT_SOLD', payload);
  }

  broadcastTick(auctionId: string, lotId: string, endTime: number) {
    this.io?.to(`auction:${auctionId}`).emit('TICK', { lotId, endTime });
  }

  startTickingLot(lot: Lot) {
    if (!lot.endsAt) return;
    const key = lot.id;
    if (this.timers.has(key)) {
      clearInterval(this.timers.get(key)!);
    }
    const timer = setInterval(() => {
      if (!lot.endsAt) return;
      const remaining = lot.endsAt.getTime() - Date.now();
      if (remaining <= 0) {
        clearInterval(timer);
        this.timers.delete(key);
      }
      this.broadcastTick(lot.auctionId, lot.id, lot.endsAt.getTime());
    }, env.tickIntervalMs);
    this.timers.set(key, timer);
  }
}
