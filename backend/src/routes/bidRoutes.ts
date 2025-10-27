import { Router } from 'express';
import { requireAuth } from '@/middleware/auth';
import { BidController } from '@/controllers/bidController';
import { BidService } from '@/services/bidService';
import { AuctionSocketGateway } from '@/ws/auctionGateway';
import rateLimit from 'express-rate-limit';

export function createBidRouter(gateway: AuctionSocketGateway) {
  const router = Router({ mergeParams: true });
  const controller = new BidController(new BidService(gateway));
  router.use(requireAuth);
  router.use(
    rateLimit({
      windowMs: 1000,
      max: 5,
      message: 'Too many bids submitted. Slow down.'
    })
  );
  router.post('/lots/:lotId/bids', (req, res, next) => controller.placeBid(req, res).catch(next));
  return router;
}
