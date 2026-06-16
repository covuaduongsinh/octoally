/**
 * STT config persistence for the web server.
 *
 * Stores a single JSON blob in the existing SQLite `settings` table under the
 * `stt_config` key. Read/written directly via getDb() (NOT through the generic
 * PUT /settings route, which drops keys not in its allow-list). API keys are
 * stored in plaintext — this is a localhost self-hosted tool, the same trust
 * level as the DB file itself.
 *
 * This is the web equivalent of the desktop's encrypted ~/.octoally/stt-config.json;
 * the two configs are intentionally independent (different machines/users).
 */

import { getDb } from '../db/index.js';
import type { VoiceCommand, BuiltinOverride } from './commands.js';

const CONFIG_KEY = 'stt_config';

export interface SttConfig {
  backend: 'local' | 'openai' | 'groq';
  openaiApiKey: string;
  groqApiKey: string;
  modelSize: string;
  language: string;
  wakePhrase: string;
  smartMatching: boolean;
  silenceTimeoutMs: number;
  maxSpeechMs: number;
  customCommands?: VoiceCommand[];
  builtinOverrides?: Record<string, BuiltinOverride>;
}

export function loadSttConfig(): Partial<SttConfig> {
  try {
    const row = getDb()
      .prepare('SELECT value FROM settings WHERE key = ?')
      .get(CONFIG_KEY) as { value: string } | undefined;
    if (!row?.value) return {};
    return JSON.parse(row.value);
  } catch {
    return {};
  }
}

export function saveSttConfig(patch: Partial<SttConfig>): void {
  const existing = loadSttConfig();
  const merged = { ...existing, ...patch };
  getDb()
    .prepare('INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value')
    .run(CONFIG_KEY, JSON.stringify(merged));
}
