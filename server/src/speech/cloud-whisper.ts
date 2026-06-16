/**
 * Cloud Whisper API transcription (OpenAI + Groq).
 * Both use the same OpenAI-compatible endpoint format.
 *
 * Copied from desktop-electron/src/speech/cloud-whisper.ts — keep in sync.
 */

import * as https from 'https';
import { createWavBuffer } from './whisper.js';

export type CloudProvider = 'openai' | 'groq';

interface ProviderConfig {
  hostname: string;
  path: string;
  model: string;
  label: string;
}

const PROVIDERS: Record<CloudProvider, ProviderConfig> = {
  openai: {
    hostname: 'api.openai.com',
    path: '/v1/audio/transcriptions',
    model: 'whisper-1',
    label: 'OpenAI',
  },
  groq: {
    hostname: 'api.groq.com',
    path: '/openai/v1/audio/transcriptions',
    model: 'whisper-large-v3-turbo',
    label: 'Groq',
  },
};

/**
 * Transcribe audio using a cloud Whisper API (OpenAI or Groq).
 * Returns the transcribed text.
 */
export function transcribeCloud(
  provider: CloudProvider,
  apiKey: string,
  samples: Float32Array,
  promptHint?: string,
  language = 'en',
): Promise<string> {
  const cfg = PROVIDERS[provider];
  if (!cfg) throw new Error(`Unknown provider: ${provider}`);

  return new Promise((resolve, reject) => {
    const wavBuffer = createWavBuffer(samples);

    // Build multipart/form-data body
    const boundary = `----OctoAllyBoundary${Date.now()}`;
    const parts: Buffer[] = [];

    // file field
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n` +
      `Content-Type: audio/wav\r\n\r\n`,
    ));
    parts.push(wavBuffer);
    parts.push(Buffer.from('\r\n'));

    // model field
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="model"\r\n\r\n` +
      `${cfg.model}\r\n`,
    ));

    // language field (speeds up processing). Omit for 'auto' so the API
    // auto-detects the spoken language.
    if (language && language !== 'auto') {
      parts.push(Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="language"\r\n\r\n` +
        `${language}\r\n`,
      ));
    }

    // prompt field (biases Whisper toward expected vocabulary)
    if (promptHint) {
      parts.push(Buffer.from(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="prompt"\r\n\r\n` +
        `${promptHint}\r\n`,
      ));
    }

    // response_format field
    parts.push(Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="response_format"\r\n\r\n` +
      `text\r\n`,
    ));

    // End boundary
    parts.push(Buffer.from(`--${boundary}--\r\n`));

    const body = Buffer.concat(parts);

    const req = https.request(
      {
        hostname: cfg.hostname,
        path: cfg.path,
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': body.length,
        },
        timeout: 30000,
      },
      (res) => {
        let data = '';
        res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
        res.on('end', () => {
          if (!res.statusCode || res.statusCode >= 400) {
            console.error(`[STT] ${cfg.label} error (${res.statusCode}): ${data}`);
            reject(new Error(`${cfg.label} API error ${res.statusCode}: ${data}`));
            return;
          }

          // response_format=text returns plain text
          resolve(data.trim());
        });
      },
    );

    req.on('error', (e) => reject(new Error(`${cfg.label} request failed: ${e.message}`)));
    req.on('timeout', () => {
      req.destroy();
      reject(new Error(`${cfg.label} request timed out`));
    });

    req.write(body);
    req.end();
  });
}
