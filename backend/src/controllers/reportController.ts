import { Request, Response } from 'express';
import { exportAuctionCsv, getHighestSale, getTeamsSummary } from '@/services/reportService';

export class ReportController {
  async highestSale(req: Request, res: Response) {
    const sale = await getHighestSale(req.params.id);
    res.json(sale);
  }

  async teamsSummary(req: Request, res: Response) {
    const summary = await getTeamsSummary(req.params.id);
    res.json(summary);
  }

  async exportCsv(req: Request, res: Response) {
    const csv = await exportAuctionCsv(req.params.id);
    res.header('Content-Type', 'text/csv');
    res.attachment(`auction-${req.params.id}.csv`);
    res.send(csv);
  }
}
