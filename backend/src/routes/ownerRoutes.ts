import { Router } from 'express';
import { requireRole } from '@/middleware/auth';
import { Role } from '@prisma/client';
import { OwnerController } from '@/controllers/ownerController';
import { AuctionService } from '@/services/auctionService';
import { AuctionSocketGateway } from '@/ws/auctionGateway';

export function createOwnerRouter(gateway: AuctionSocketGateway) {
  const router = Router();
  const controller = new OwnerController(new AuctionService(gateway));
  router.use(requireRole(Role.OWNER));
  router.get('/auctions/:id/team', (req, res, next) => controller.myTeam(req, res).catch(next));
  router.get('/auctions/:id/active-lot', (req, res, next) => controller.activeLot(req, res).catch(next));
  return router;
}
