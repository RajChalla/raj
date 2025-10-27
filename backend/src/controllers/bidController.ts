import { Request, Response } from 'express';
import createHttpError from 'http-errors';
import { BidService } from '@/services/bidService';
import { prisma } from '@/config/prisma';
import { Role } from '@prisma/client';

export class BidController {
  constructor(private bidService: BidService) {}

  async placeBid(req: Request, res: Response) {
    const { amount, teamId: bodyTeamId } = req.body as { amount: number; teamId?: string };
    if (!req.user) {
      throw new createHttpError.Unauthorized();
    }
    const lot = await prisma.lot.findUnique({ where: { id: req.params.lotId } });
    if (!lot) {
      throw new createHttpError.NotFound('Lot not found');
    }
    let teamId = bodyTeamId;
    if (req.user.role === Role.OWNER) {
      const team = await prisma.team.findFirst({ where: { ownerId: req.user.id, auctionId: lot.auctionId } });
      if (!team) {
        throw new createHttpError.BadRequest('Owner has no team in auction');
      }
      teamId = team.id;
    }
    if (!teamId) {
      throw new createHttpError.BadRequest('teamId required');
    }
    const bid = await this.bidService.placeBid(req.params.lotId, teamId, amount);
    res.status(201).json(bid);
  }
}
