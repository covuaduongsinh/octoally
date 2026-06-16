/**
 * Renderer-side microphone capture via the Web Audio API.
 *
 * Used wherever the renderer owns audio capture: the Windows desktop (the main
 * process has no native arecord/sox path) and the web browser (no main process
 * at all). Captures 16kHz mono Float32 PCM and streams it via `sttPushAudio`,
 * which routes to the desktop IPC (`stt_push_audio`) or the /api/stt WebSocket
 * binary stream — both feed the same VAD + cloud transcription pipeline.
 */
import { sttPushAudio } from './stt-client';

let stream: MediaStream | null = null;
let ctx: AudioContext | null = null;
let source: MediaStreamAudioSourceNode | null = null;
let processor: ScriptProcessorNode | null = null;
let sink: GainNode | null = null;

/** True while the mic is being captured. */
export function isCapturing(): boolean {
  return ctx !== null;
}

/**
 * Start capturing the microphone and streaming PCM to the main process.
 * Throws if mic permission is denied or audio initialization fails — the
 * caller is responsible for surfacing the error and reverting mic state.
 */
export async function startMicCapture(): Promise<void> {
  if (ctx) return; // already capturing

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  // Force a 16kHz context so frames arrive at the rate whisper/VAD expect —
  // Chromium resamples the mic stream into the context automatically.
  ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.resume();

  source = ctx.createMediaStreamSource(stream);
  processor = ctx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0); // Float32 @16kHz
    // Clone — the underlying buffer is reused by the audio thread.
    sttPushAudio(new Float32Array(input));
  };

  // ScriptProcessorNode only fires while connected to a destination. Route it
  // through a muted gain node so the mic isn't echoed back to the speakers.
  sink = ctx.createGain();
  sink.gain.value = 0;
  source.connect(processor);
  processor.connect(sink);
  sink.connect(ctx.destination);
}

/** Stop capturing the microphone and release all audio resources. */
export function stopMicCapture(): void {
  try {
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
    }
  } catch { /* ignore */ }
  try { source?.disconnect(); } catch { /* ignore */ }
  try { sink?.disconnect(); } catch { /* ignore */ }
  try { stream?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
  try { void ctx?.close(); } catch { /* ignore */ }
  processor = null;
  source = null;
  sink = null;
  stream = null;
  ctx = null;
}
