import { useState } from 'react';
import { PageLayout } from '../components/PageLayout';
import { api } from '../services/api';

export function ReportsPage() {
  const [auctionId, setAuctionId] = useState('');
  const [highest, setHighest] = useState<any>(null);
  const [teams, setTeams] = useState<any[]>([]);

  const loadReports = async () => {
    const [highestRes, teamRes] = await Promise.all([
      api.get(`/auctions/${auctionId}/reports/highest-sale`),
      api.get(`/auctions/${auctionId}/reports/teams-summary`)
    ]);
    setHighest(highestRes.data);
    setTeams(teamRes.data);
  };

  const exportCsv = async () => {
    const res = await api.get(`/auctions/${auctionId}/export.csv`, { responseType: 'blob' });
    const url = window.URL.createObjectURL(res.data);
    const link = document.createElement('a');
    link.href = url;
    link.download = `auction-${auctionId}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <PageLayout>
      <section className="card">
        <h2>Auction Reports</h2>
        <input value={auctionId} onChange={(e) => setAuctionId(e.target.value)} placeholder="Auction ID" />
        <div className="actions">
          <button onClick={loadReports} disabled={!auctionId}>
            Load
          </button>
          <button onClick={exportCsv} disabled={!auctionId}>
            Export CSV
          </button>
        </div>
        {highest && (
          <div className="report-block">
            <h3>Highest Sale</h3>
            <p>
              {highest.player?.name} ({highest.player?.rating}) sold for {highest.soldPrice?.toLocaleString()} (base {highest.basePrice})
            </p>
          </div>
        )}
        {teams.length > 0 && (
          <div className="report-block">
            <h3>Teams</h3>
            <table>
              <thead>
                <tr>
                  <th>Team</th>
                  <th>Players</th>
                  <th>Spent</th>
                  <th>Base</th>
                  <th>Delta</th>
                  <th>Remaining</th>
                </tr>
              </thead>
              <tbody>
                {teams.map((team) => (
                  <tr key={team.teamId}>
                    <td>{team.teamName}</td>
                    <td>
                      <ul>
                        {team.players.map((player: any) => (
                          <li key={player.playerId}>
                            {player.name}: {player.soldPrice?.toLocaleString()} (base {player.basePrice.toLocaleString()})
                          </li>
                        ))}
                      </ul>
                    </td>
                    <td>{team.totalSpent.toLocaleString()}</td>
                    <td>{team.totalBase.toLocaleString()}</td>
                    <td>{team.delta.toLocaleString()}</td>
                    <td>{team.remainingBudget.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </PageLayout>
  );
}
