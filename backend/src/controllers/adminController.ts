import { Request, Response } from 'express';
import { AuctionService } from '@/services/auctionService';
import { importPlayers } from '@/services/playerService';
import { updateBasePriceTiers } from '@/services/settingsService';
import { z } from 'zod';

const tiersSchema = z.array(
  z.object({
    min: z.number(),
    max: z.number(),
    price: z.number()
  })
);

export class AdminController {
  constructor(private auctionService: AuctionService) {}

  async createAuction(req: Request, res: Response) {
    const { name } = req.body;
    const auction = await this.auctionService.createAuction(name, req.user!.id);
    res.status(201).json(auction);
  }

  async importPlayers(req: Request, res: Response) {
    await importPlayers(req.body.players ?? [], req.user!.id);
    res.status(204).send();
  }

  async updateBasePrice(req: Request, res: Response) {
    const tiers = tiersSchema.parse(req.body.tiers);
    await updateBasePriceTiers(tiers, req.user!.id);
    res.status(204).send();
  }

  async setTeams(req: Request, res: Response) {
    const { teams } = req.body as { teams: { name: string; ownerId?: string; budget: number }[] };
    const result = await this.auctionService.setTeams(req.params.id, teams, req.user!.id);
    res.json(result);
  }

  async startAuction(req: Request, res: Response) {
    const auction = await this.auctionService.startAuction(req.params.id, req.user!.id);
    res.json(auction);
  }

  async pauseAuction(req: Request, res: Response) {
    const auction = await this.auctionService.pauseAuction(req.params.id, req.user!.id);
    res.json(auction);
  }

  async resumeAuction(req: Request, res: Response) {
    const auction = await this.auctionService.resumeAuction(req.params.id, req.user!.id);
    res.json(auction);
  }

  async completeAuction(req: Request, res: Response) {
    const { override = false, reason } = req.body ?? {};
    const auction = await this.auctionService.completeAuction(req.params.id, req.user!.id, override, reason);
    res.json(auction);
  }

  async queueLots(req: Request, res: Response) {
    const { playerIds } = req.body as { playerIds: string[] };
    const lots = await this.auctionService.queueLots(req.params.id, playerIds, req.user!.id);
    res.json(lots);
  }

  async activateNextLot(req: Request, res: Response) {
    const { durationSeconds = 60 } = req.body ?? {};
    const lot = await this.auctionService.activateNextLot(req.params.id, durationSeconds);
    res.json(lot);
  }

  async markLotUnsold(req: Request, res: Response) {
    const lot = await this.auctionService.markLotUnsold(req.params.lotId, req.user!.id);
    res.json(lot);
  }

  async markLotSold(req: Request, res: Response) {
    await this.auctionService.closeLotAsSold(req.params.lotId, req.user!.id);
    res.status(204).send();
  }
}
