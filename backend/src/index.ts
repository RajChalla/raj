import http from 'http';
import { Server } from 'socket.io';
import { createApp } from '@/app';
import { env } from '@/config/env';
import { AuctionSocketGateway } from '@/ws/auctionGateway';
import { prisma } from '@/config/prisma';

async function bootstrap() {
  const gateway = new AuctionSocketGateway();
  const app = createApp(gateway);
  const server = http.createServer(app);
  const io = new Server(server, {
    cors: { origin: true, credentials: true }
  });
  gateway.setServer(io);

  io.on('connection', (socket) => {
    const { auctionId } = socket.handshake.query;
    if (auctionId && typeof auctionId === 'string') {
      socket.join(`auction:${auctionId}`);
    }
  });

  const port = env.port;
  server.listen(port, () => {
    console.log(`Server listening on port ${port}`);
  });
}

bootstrap().catch(async (err) => {
  console.error(err);
  await prisma.$disconnect();
  process.exit(1);
});
