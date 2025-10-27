import { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

export function useAuctionSocket(auctionId?: string) {
  const socketRef = useRef<Socket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!auctionId) return;
    const socket = io('/', { query: { auctionId } });
    socketRef.current = socket;
    socket.on('connect', () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));
    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [auctionId]);

  return { socket: socketRef.current, connected };
}
