/**
 * Transport-aware STT client.
 *
 * The dashboard runs in two environments:
 *  - Desktop (Electron/Tauri): the STT pipeline lives in the desktop main
 *    process and is driven over IPC — delegate to ./tauri.
 *  - Web (browser): there is no main process, so we drive a server-side STT
 *    pipeline over a WebSocket at /api/stt. Control messages are request/
 *    response correlated by id; mic audio is sent as raw binary frames.
 *
 * speech.ts and mic-capture.ts use these helpers instead of ./tauri directly,
 * so the same store/UI works in both environments with no behaviour change.
 */

import { isDesktop, invoke as desktopInvoke, listen as desktopListen } from './tauri';

// getUserMedia requires a secure context — works on https and http://localhost,
// but NOT on plain-http LAN-IP access. Hide voice UI cleanly in that case.
const webSttSupported =
  !isDesktop &&
  typeof navigator !== 'undefined' &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof window !== 'undefined' &&
  window.isSecureContext === true;

/** True when STT is usable in this environment (desktop, or a secure browser). */
export const sttAvailable = isDesktop || webSttSupported;

/**
 * True when the browser CAN do voice but the page isn't a secure context, so
 * it's disabled. Lets the UI show a "use https/localhost" hint.
 */
export const sttInsecureContext =
  !isDesktop &&
  typeof navigator !== 'undefined' &&
  !!navigator.mediaDevices?.getUserMedia &&
  typeof window !== 'undefined' &&
  window.isSecureContext !== true;

// ---------------------------------------------------------------------------
// Web WebSocket transport
// ---------------------------------------------------------------------------

interface PendingInvoke {
  resolve: (value: any) => void;
  reject: (err: any) => void;
  timer: ReturnType<typeof setTimeout>;
}

let ws: WebSocket | null = null;
let nextId = 1;
const pending = new Map<number, PendingInvoke>();
const handlers = new Map<string, Set<(payload: any) => void>>();
const controlQueue: string[] = [];
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function wsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/api/stt`;
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  try {
    ws = new WebSocket(wsUrl());
  } catch {
    scheduleReconnect();
    return;
  }
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    // Flush any control messages queued while connecting.
    for (const msg of controlQueue.splice(0)) {
      try { ws!.send(msg); } catch { /* ignore */ }
    }
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return; // server only sends text
    let msg: any;
    try { msg = JSON.parse(ev.data); } catch { return; }

    if (msg.kind === 'result') {
      const p = pending.get(msg.id);
      if (p) { clearTimeout(p.timer); pending.delete(msg.id); p.resolve(msg.result); }
    } else if (msg.kind === 'error') {
      const p = pending.get(msg.id);
      if (p) { clearTimeout(p.timer); pending.delete(msg.id); p.reject(new Error(msg.error)); }
    } else if (msg.kind === 'event') {
      const set = handlers.get(msg.channel);
      if (set) for (const fn of set) { try { fn(msg.data); } catch { /* ignore */ } }
    }
  };

  ws.onclose = () => {
    ws = null;
    scheduleReconnect();
  };

  ws.onerror = () => {
    try { ws?.close(); } catch { /* ignore */ }
  };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    // Reconnect only if something still cares (active listeners or in-flight calls).
    if (handlers.size > 0 || pending.size > 0) connect();
  }, 3000);
}

function ensureWs() {
  if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) connect();
}

function sendControl(obj: unknown) {
  const data = JSON.stringify(obj);
  ensureWs();
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(data); return; } catch { /* fall through to queue */ }
  }
  controlQueue.push(data);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Invoke an STT command. Desktop → IPC; web → /api/stt WebSocket. */
export async function sttInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (isDesktop) return desktopInvoke<T>(cmd, args);

  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`STT request timed out: ${cmd}`));
    }, 15000);
    pending.set(id, { resolve, reject, timer });
    sendControl({ kind: 'invoke', id, cmd, args: args ?? {} });
  });
}

/** Subscribe to an STT event. Returns an unlisten function. */
export async function sttListen<T>(event: string, handler: (payload: T) => void): Promise<() => void> {
  if (isDesktop) return desktopListen<T>(event, handler);

  let set = handlers.get(event);
  if (!set) { set = new Set(); handlers.set(event, set); }
  set.add(handler as (p: any) => void);
  ensureWs();

  return () => {
    const s = handlers.get(event);
    if (s) s.delete(handler as (p: any) => void);
  };
}

/** Stream a frame of 16kHz mono Float32 PCM to the STT backend (fire-and-forget). */
export function sttPushAudio(samples: Float32Array): void {
  if (isDesktop) {
    void desktopInvoke('stt_push_audio', { samples }).catch(() => {});
    return;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    // Copy the exact bytes (the view may be a sub-range of a larger buffer).
    try {
      ws.send(samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength));
    } catch { /* dropped frame — fine for realtime audio */ }
  }
  // If the socket isn't open we simply drop the frame.
}
