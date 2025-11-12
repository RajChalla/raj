import { createHash } from 'crypto';

function createAcceptValue(key) {
  return createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
}

function encodeMessage(message) {
  const json = JSON.stringify(message);
  const length = Buffer.byteLength(json);
  const buffer = Buffer.alloc(length + 2);
  buffer[0] = 0x81;
  buffer[1] = length;
  buffer.write(json, 2);
  return buffer;
}

export class AuctionSocketServer {
  constructor() {
    this.sockets = new Map();
  }

  attach(server) {
    server.on('upgrade', (req, socket) => {
      if (req.headers['upgrade'] !== 'websocket') {
        socket.destroy();
        return;
      }
      const key = req.headers['sec-websocket-key'];
      const accept = createAcceptValue(key);
      const auctionId = new URL(req.url, 'http://localhost').searchParams.get('auctionId');
      socket.write(
        'HTTP/1.1 101 Switching Protocols\r\n' +
          'Upgrade: websocket\r\n' +
          'Connection: Upgrade\r\n' +
          `Sec-WebSocket-Accept: ${accept}\r\n` +
          '\r\n'
      );
      socket.on('close', () => {
        for (const [id, set] of this.sockets.entries()) {
          if (set.has(socket)) {
            set.delete(socket);
          }
        }
      });
      if (!this.sockets.has(auctionId)) {
        this.sockets.set(auctionId, new Set());
      }
      this.sockets.get(auctionId).add(socket);
    });
  }

  broadcast(auctionId, event, payload) {
    const sockets = this.sockets.get(auctionId);
    if (!sockets) return;
    const frame = encodeMessage({ event, payload });
    for (const socket of sockets) {
      socket.write(frame);
    }
  }
}
