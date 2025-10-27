import { FormEvent, useEffect, useState } from 'react';
import { PageLayout } from '../components/PageLayout';
import { api } from '../services/api';
import { useAuctionSocket } from '../hooks/useAuctionSocket';

interface TeamForm {
  name: string;
  ownerId: string;
  budget: number;
}

export function AdminDashboard() {
  const [auctionId, setAuctionId] = useState<string>('');
  const [auctionName, setAuctionName] = useState('');
  const [teams, setTeams] = useState<TeamForm[]>([{ name: '', ownerId: '', budget: 100000 }]);
  const [playerPayload, setPlayerPayload] = useState('');
  const [playerIds, setPlayerIds] = useState('');
  const [lotIdAction, setLotIdAction] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [override, setOverride] = useState(false);
  const [overrideReason, setOverrideReason] = useState('');

  const { socket } = useAuctionSocket(auctionId);

  useEffect(() => {
    if (!socket) return;
    const appendLog = (message: string) => setLogs((prev) => [message, ...prev.slice(0, 24)]);
    socket.on('LOT_ACTIVATED', (lot) => appendLog(`Lot activated for ${lot.player?.name}`));
    socket.on('BID_PLACED', (bid) => appendLog(`Bid ${bid.amount} by ${bid.teamId}`));
    socket.on('LOT_SOLD', (payload) => appendLog(`Lot ${payload.lotId} sold at ${payload.soldPrice ?? 'N/A'}`));
    socket.on('AUCTION_STATUS_CHANGED', (payload) => appendLog(`Auction status ${payload.status}`));
    socket.on('TICK', (payload) => appendLog(`Tick ${new Date(payload.endTime).toLocaleTimeString()}`));
    return () => {
      socket.off('LOT_ACTIVATED');
      socket.off('BID_PLACED');
      socket.off('LOT_SOLD');
      socket.off('AUCTION_STATUS_CHANGED');
      socket.off('TICK');
    };
  }, [socket]);

  const handleCreateAuction = async (e: FormEvent) => {
    e.preventDefault();
    const res = await api.post('/auctions', { name: auctionName });
    setAuctionId(res.data.id);
  };

  const handleTeamsSubmit = async (e: FormEvent) => {
    e.preventDefault();
    await api.post(`/auctions/${auctionId}/teams`, { teams });
  };

  const handleImportPlayers = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = playerPayload.trim();
    try {
      if (trimmed.startsWith('[')) {
        const payload = JSON.parse(trimmed);
        await api.post('/players/import', { players: payload });
      } else {
        await api.post('/players/import', { players: trimmed });
      }
      setPlayerPayload('');
    } catch (err) {
      alert('Invalid player payload');
    }
  };

  const handleQueuePlayers = async (e: FormEvent) => {
    e.preventDefault();
    const ids = playerIds.split(/\s|,/).filter(Boolean);
    await api.post(`/auctions/${auctionId}/lots`, { playerIds: ids });
    setPlayerIds('');
  };

  const handleStart = () => api.patch(`/auctions/${auctionId}/start`);
  const handlePause = () => api.patch(`/auctions/${auctionId}/pause`);
  const handleResume = () => api.patch(`/auctions/${auctionId}/resume`);
  const handleComplete = () => api.patch(`/auctions/${auctionId}/complete`, { override, reason: overrideReason });
  const handleActivate = () => api.post(`/auctions/${auctionId}/lots/activate`, { durationSeconds: 60 });
  const handleMarkUnsold = () => api.patch(`/lots/${lotIdAction}/unsold`);
  const handleMarkSold = () => api.patch(`/lots/${lotIdAction}/sold`);

  return (
    <PageLayout>
      <div className="grid">
        <section className="card">
          <h2>Create Auction</h2>
          <form onSubmit={handleCreateAuction} className="form">
            <input value={auctionName} onChange={(e) => setAuctionName(e.target.value)} placeholder="Auction name" required />
            <button type="submit">Create</button>
          </form>
          {auctionId && <p>Current auction: {auctionId}</p>}
        </section>

        <section className="card">
          <h2>Teams & Budgets</h2>
          <form onSubmit={handleTeamsSubmit} className="form">
            {teams.map((team, idx) => (
              <div key={idx} className="team-row">
                <input
                  value={team.name}
                  onChange={(e) => setTeams((prev) => prev.map((t, i) => (i === idx ? { ...t, name: e.target.value } : t)))}
                  placeholder="Team name"
                  required
                />
                <input
                  value={team.ownerId}
                  onChange={(e) => setTeams((prev) => prev.map((t, i) => (i === idx ? { ...t, ownerId: e.target.value } : t)))}
                  placeholder="Owner user id"
                  required
                />
                <input
                  type="number"
                  value={team.budget}
                  onChange={(e) => setTeams((prev) => prev.map((t, i) => (i === idx ? { ...t, budget: Number(e.target.value) } : t)))}
                  placeholder="Budget"
                  required
                />
              </div>
            ))}
            <div className="actions">
              <button type="button" onClick={() => setTeams((prev) => [...prev, { name: '', ownerId: '', budget: 100000 }])}>
                Add team
              </button>
              <button type="submit" disabled={!auctionId}>
                Save teams
              </button>
            </div>
          </form>
        </section>

        <section className="card">
          <h2>Import Players</h2>
          <form onSubmit={handleImportPlayers} className="form">
            <textarea
              value={playerPayload}
              onChange={(e) => setPlayerPayload(e.target.value)}
              placeholder='[{"id":"","name":"","position":"","club":"","nation":"","rating":90,"rarity":"Gold Rare","gender":"Men"}]'
              rows={6}
              required
            />
            <button type="submit">Import</button>
          </form>
        </section>

        <section className="card">
          <h2>Queue Players</h2>
          <form onSubmit={handleQueuePlayers} className="form">
            <textarea
              value={playerIds}
              onChange={(e) => setPlayerIds(e.target.value)}
              placeholder="Enter player IDs separated by space or comma"
              rows={4}
              required
            />
            <button type="submit" disabled={!auctionId}>
              Queue lots
            </button>
          </form>
        </section>

        <section className="card">
          <h2>Auction Controls</h2>
          <div className="actions">
            <button onClick={handleStart} disabled={!auctionId}>
              Start
            </button>
            <button onClick={handlePause} disabled={!auctionId}>
              Pause
            </button>
            <button onClick={handleResume} disabled={!auctionId}>
              Resume
            </button>
            <button onClick={handleActivate} disabled={!auctionId}>
              Activate next lot
            </button>
          </div>
          <div className="form">
            <label>
              Lot ID
              <input value={lotIdAction} onChange={(e) => setLotIdAction(e.target.value)} placeholder="Lot id" />
            </label>
            <div className="actions">
              <button onClick={handleMarkSold} type="button">
                Mark sold
              </button>
              <button onClick={handleMarkUnsold} type="button">
                Mark unsold
              </button>
            </div>
          </div>
          <div className="form">
            <label>
              Override completion
              <input type="checkbox" checked={override} onChange={(e) => setOverride(e.target.checked)} />
            </label>
            <input
              value={overrideReason}
              onChange={(e) => setOverrideReason(e.target.value)}
              placeholder="Reason"
              disabled={!override}
            />
            <button onClick={handleComplete} type="button" disabled={!auctionId}>
              Complete auction
            </button>
          </div>
        </section>

        <section className="card">
          <h2>Live Activity</h2>
          <div className="log">
            {logs.map((log, idx) => (
              <p key={idx}>{log}</p>
            ))}
          </div>
        </section>
      </div>
    </PageLayout>
  );
}
