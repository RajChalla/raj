import { Request, Response } from 'express';
import createHttpError from 'http-errors';
import { prisma } from '@/config/prisma';
import { AuctionService } from '@/services/auctionService';
import { AuctionSocketGateway } from '@/ws/auctionGateway';

export class OwnerController {
  constructor(private auctionService: AuctionService) {}

  async myTeam(req: Request, res: Response) {
    const team = await prisma.team.findFirst({
      where: { auctionId: req.params.id, ownerId: req.user!.id },
      include: {
        sales: {
          include: { player: true }
        }
      }
    });
    if (!team) {
      throw new createHttpError.NotFound('Team not found');
    }
    res.json(team);
  }

  async activeLot(req: Request, res: Response) {
    const team = await prisma.team.findFirst({ where: { auctionId: req.params.id, ownerId: req.user!.id } });
    if (!team) {
      throw new createHttpError.NotFound('Team not found');
    }
    const lot = await this.auctionService.getActiveLot(req.params.id);
    res.json(lot);
  }
}
