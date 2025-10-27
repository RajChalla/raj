import { prisma } from '@/config/prisma';
import { stringify } from 'csv-stringify/sync';

export async function getHighestSale(auctionId: string) {
  const sale = await prisma.sale.findFirst({
    where: { auctionId, soldPrice: { not: null } },
    orderBy: { soldPrice: 'desc' },
    include: { player: true, team: true }
  });
  return sale;
}

export async function getTeamsSummary(auctionId: string) {
  const teams = await prisma.team.findMany({
    where: { auctionId },
    include: {
      sales: {
        include: { player: true },
        orderBy: { createdAt: 'asc' }
      }
    }
  });
  return teams.map((team) => {
    const players = team.sales
      .filter((sale) => sale.soldPrice !== null)
      .map((sale) => ({
        playerId: sale.playerId,
        name: sale.player?.name,
        soldPrice: sale.soldPrice,
        basePrice: sale.basePrice
      }));
    const totalSpent = players.reduce((sum, sale) => sum + (sale.soldPrice ?? 0), 0);
    const totalBase = players.reduce((sum, sale) => sum + sale.basePrice, 0);
    return {
      teamId: team.id,
      teamName: team.name,
      players,
      totalSpent,
      totalBase,
      delta: totalSpent - totalBase,
      remainingBudget: team.remainingBudget
    };
  });
}

export async function exportAuctionCsv(auctionId: string) {
  const teams = await getTeamsSummary(auctionId);
  const rows: (string | number)[][] = [
    ['Team', 'Player', 'Sold Price', 'Base Price', 'Delta', 'Remaining Budget']
  ];
  teams.forEach((team) => {
    if (team.players.length === 0) {
      rows.push([team.teamName, '-', 0, 0, 0, team.remainingBudget]);
    }
    team.players.forEach((player) => {
      rows.push([
        team.teamName,
        player.name ?? player.playerId,
        player.soldPrice ?? 0,
        player.basePrice,
        (player.soldPrice ?? 0) - player.basePrice,
        team.remainingBudget
      ]);
    });
  });
  return stringify(rows);
}
