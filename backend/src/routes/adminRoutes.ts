import { Router } from 'express';
import { requireRole } from '@/middleware/auth';
import { Role } from '@prisma/client';
import { AdminController } from '@/controllers/adminController';
import { AuctionService } from '@/services/auctionService';
import { AuctionSocketGateway } from '@/ws/auctionGateway';

export function createAdminRouter(gateway: AuctionSocketGateway) {
  const router = Router();
  const controller = new AdminController(new AuctionService(gateway));

  router.use(requireRole(Role.ADMIN));
  router.post('/auctions', (req, res, next) => controller.createAuction(req, res).catch(next));
  router.post('/players/import', (req, res, next) => controller.importPlayers(req, res).catch(next));
  router.patch('/settings/basePriceTiers', (req, res, next) => controller.updateBasePrice(req, res).catch(next));
  router.post('/auctions/:id/teams', (req, res, next) => controller.setTeams(req, res).catch(next));
  router.patch('/auctions/:id/start', (req, res, next) => controller.startAuction(req, res).catch(next));
  router.patch('/auctions/:id/pause', (req, res, next) => controller.pauseAuction(req, res).catch(next));
  router.patch('/auctions/:id/resume', (req, res, next) => controller.resumeAuction(req, res).catch(next));
  router.patch('/auctions/:id/complete', (req, res, next) => controller.completeAuction(req, res).catch(next));
  router.post('/auctions/:id/lots', (req, res, next) => controller.queueLots(req, res).catch(next));
  router.post('/auctions/:id/lots/activate', (req, res, next) => controller.activateNextLot(req, res).catch(next));
  router.patch('/lots/:lotId/unsold', (req, res, next) => controller.markLotUnsold(req, res).catch(next));
  router.patch('/lots/:lotId/sold', (req, res, next) => controller.markLotSold(req, res).catch(next));

  return router;
}
