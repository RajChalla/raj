import { useEffect, useState } from 'react';
import { PageLayout } from '../components/PageLayout';
import { api } from '../services/api';
import { useAuctionSocket } from '../hooks/useAuctionSocket';

interface PlayerSale {
  player: { name: string };
  soldPrice: number | null;
  basePrice: number;
}

interface TeamData {
  id: string;
  name: string;
  remainingBudget: number;
  rosterCount: number;
  auctionId: string;
  sales: PlayerSale[];
}

interface LotData {
  id: string;
  reservePrice: number;
  player: { name: string; basePrice: number; rating: number; position: string };
  currentBid?: { amount: number; teamId: string };
  endsAt?: string;
}

export function OwnerDashboard() {
  const [auctionId, setAuctionId] = useState('');
  const [team, setTeam] = useState<TeamData | null>(null);
  const [lot, setLot] = useState<LotData | null>(null);
  const [bidAmount, setBidAmount] = useState(0);
  const [history, setHistory] = useState<string[]>([]);
  const { socket } = useAuctionSocket(auctionId);

  useEffect(() => {
    if (!socket) return;
    const push = (entry: string) => setHistory((prev) => [entry, ...prev.slice(0, 19)]);
    socket.on('LOT_ACTIVATED', (data) => {
      setLot(data);
      push(`Lot activated: ${data.player?.name}`);
    });
    socket.on('BID_PLACED', (data) => {
      push(`Bid ${data.amount} by ${data.teamId}`);
      if (data.lotId === lot?.id) {
        setLot((prev) => (prev ? { ...prev, currentBid: { amount: data.amount, teamId: data.teamId } } : prev));
      }
      if (team && data.teamId === team.id) {
        setTeam((prev) => (prev ? { ...prev, remainingBudget: data.remainingBudget } : prev));
      }
    });
    socket.on('LOT_SOLD', (payload) => push(`Lot sold for ${payload.soldPrice ?? 'N/A'}`));
    socket.on('TICK', (payload) => push(`Ends at ${new Date(payload.endTime).toLocaleTimeString()}`));
    return () => {
      socket.off('LOT_ACTIVATED');
      socket.off('BID_PLACED');
      socket.off('LOT_SOLD');
      socket.off('TICK');
    };
  }, [socket, lot?.id, team]);

  useEffect(() => {
    if (!auctionId) return;
    api
      .get<TeamData>(`/auctions/${auctionId}/team`)
      .then((res) => {
        setTeam(res.data);
        return api.get<LotData | null>(`/auctions/${auctionId}/active-lot`);
      })
      .then((res) => {
        setLot(res.data);
      })
      .catch(() => {
        setTeam(null);
        setLot(null);
      });
  }, [auctionId]);

  const increment = (value: number) => setBidAmount((prev) => Math.max(0, prev + value));

  const placeBid = async () => {
    if (!lot) return;
    await api.post(`/lots/${lot.id}/bids`, { amount: bidAmount });
    setBidAmount(0);
  };

  return (
    <PageLayout>
      <div className="grid">
        <section className="card">
          <h2>Select Auction</h2>
          <input value={auctionId} onChange={(e) => setAuctionId(e.target.value)} placeholder="Auction ID" />
          {team && (
            <div className="team-info">
              <p>Team: {team.name}</p>
              <p>Remaining budget: {team.remainingBudget.toLocaleString()}</p>
              <p>Roster count: {team.rosterCount}</p>
            </div>
          )}
        </section>

        {lot && (
          <section className="card">
            <h2>Active Lot</h2>
            <p>{lot.player.name}</p>
            <p>
              Rating {lot.player.rating} • Position {lot.player.position}
            </p>
            <p>Reserve: {lot.reservePrice.toLocaleString()}</p>
            <p>Current bid: {lot.currentBid?.amount?.toLocaleString() ?? 'None'}</p>
            <div className="actions">
              {[1000, 5000, 10000].map((inc) => (
                <button key={inc} onClick={() => increment(inc)}>
                  +{inc.toLocaleString()}
                </button>
              ))}
              <button onClick={() => increment(-(bidAmount || 0))}>Clear</button>
            </div>
            <input
              type="number"
              value={bidAmount}
              onChange={(e) => setBidAmount(Number(e.target.value))}
              placeholder="Bid amount"
            />
            <button onClick={placeBid} disabled={!bidAmount}>
              Submit Bid
            </button>
          </section>
        )}

        <section className="card">
          <h2>Recent Activity</h2>
          <div className="log">
            {history.map((entry, idx) => (
              <p key={idx}>{entry}</p>
            ))}
          </div>
        </section>

        {team && (
          <section className="card">
            <h2>Acquired Players</h2>
            <ul>
              {team.sales
                .filter((sale) => sale.soldPrice)
                .map((sale, idx) => (
                  <li key={idx}>
                    {sale.player.name} – {sale.soldPrice?.toLocaleString()} (base {sale.basePrice.toLocaleString()})
                  </li>
                ))}
            </ul>
          </section>
        )}
      </div>
    </PageLayout>
  );
}
