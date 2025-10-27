import { Router } from 'express';
import { requireAuth } from '@/middleware/auth';
import { ReportController } from '@/controllers/reportController';

const router = Router();
const controller = new ReportController();

router.use(requireAuth);
router.get('/auctions/:id/reports/highest-sale', (req, res, next) => controller.highestSale(req, res).catch(next));
router.get('/auctions/:id/reports/teams-summary', (req, res, next) => controller.teamsSummary(req, res).catch(next));
router.get('/auctions/:id/export.csv', (req, res, next) => controller.exportCsv(req, res).catch(next));

export default router;
