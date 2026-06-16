import { FastifyPluginAsync } from 'fastify';
import { SttSession } from '../speech/stt-session.js';

/**
 * Speech-to-text WebSocket route — the web equivalent of the desktop Electron
 * IPC bridge. Connect to /api/stt to drive the server-side STT pipeline.
 *
 * Protocol:
 *  - TEXT frames carry control JSON:
 *      client → { kind: 'invoke', id, cmd, args }
 *      server → { kind: 'result', id, result } | { kind: 'error', id, error }
 *               { kind: 'event', channel: 'stt://...', data }
 *  - BINARY frames carry raw Float32 PCM (16kHz mono) — the mic audio stream.
 *
 * One SttSession per connection (per browser tab); each owns its own VAD/config.
 *
 * Keep the speech pipeline (server/src/speech/) in sync with
 * desktop-electron/src/speech/ — it is a deliberate copy.
 */

function isLoopbackAddr(addr?: string | null): boolean {
  if (!addr) return false;
  return (
    addr === '127.0.0.1' ||
    addr === '::1' ||
    addr === '::ffff:127.0.0.1' ||
    addr.startsWith('127.')
  );
}

export const sttRoutes: FastifyPluginAsync = async (app) => {
  app.get('/stt', { websocket: true }, (socket, req) => {
    const loopback = isLoopbackAddr(req.socket?.remoteAddress);

    const session = new SttSession(
      (channel, data) => {
        try {
          socket.send(JSON.stringify({ kind: 'event', channel, data }));
        } catch { /* socket closing */ }
      },
      { isLoopback: loopback },
    );

    socket.on('message', (raw: Buffer, isBinary: boolean) => {
      // Binary frame = a chunk of 16kHz Float32 PCM from the renderer mic.
      if (isBinary) {
        const buf = raw as Buffer;
        // Copy out — Buffer memory is pooled and may not be 4-byte aligned.
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        if (ab.byteLength % 4 === 0) {
          session.feedSamples(new Float32Array(ab));
        }
        return;
      }

      // Text frame = a control message.
      let msg: any;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return;
      }
      if (!msg || msg.kind !== 'invoke') return;

      Promise.resolve()
        .then(() => session.dispatch(msg.cmd, msg.args || {}))
        .then((result) => {
          try { socket.send(JSON.stringify({ kind: 'result', id: msg.id, result })); } catch { /* closing */ }
        })
        .catch((err) => {
          try { socket.send(JSON.stringify({ kind: 'error', id: msg.id, error: err?.message ?? String(err) })); } catch { /* closing */ }
        });
    });

    socket.on('close', () => session.dispose());
  });
};
