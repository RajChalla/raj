import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import rateLimit from 'express-rate-limit';
import authRoutes from '@/routes/authRoutes';
import reportRoutes from '@/routes/reportRoutes';
import { createAdminRouter } from '@/routes/adminRoutes';
import { createBidRouter } from '@/routes/bidRoutes';
import { authenticate, requireAuth } from '@/middleware/auth';
import { errorHandler } from '@/middleware/errorHandler';
import { AuctionSocketGateway } from '@/ws/auctionGateway';
import { createOwnerRouter } from '@/routes/ownerRoutes';

export function createApp(gateway: AuctionSocketGateway) {
  const app = express();
  const limiter = rateLimit({ windowMs: 1000, max: 50 });
  app.use(limiter);
  app.use(helmet());
  app.use(cors({ origin: true, credentials: true }));
  app.use(express.json());
  app.use(cookieParser());
  app.use(authenticate);
  app.use('/api/auth', authRoutes);
  app.use('/api', requireAuth, reportRoutes);
  app.use('/api', requireAuth, createOwnerRouter(gateway));
  app.use('/api', requireAuth, createBidRouter(gateway));
  app.use('/api', requireAuth, createAdminRouter(gateway));

  app.use(errorHandler);
  return app;
}
