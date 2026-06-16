/**
 * Per-connection speech-to-text state machine for the web server.
 *
 * This is the web equivalent of the Electron main process speech module
 * (desktop-electron/src/speech/index.ts). The browser captures the mic via Web
 * Audio and streams 16kHz PCM here over the /api/stt WebSocket; this class runs
 * VAD + cloud transcription + voice-command matching and emits the same
 * `stt://*` events back, so the renderer's speech store needs no behaviour change.
 *
 * One instance per WebSocket connection (per browser tab) — all state is on the
 * instance, never module-global.
 *
 * Phase 1 = cloud backends only (OpenAI/Groq). Local Whisper + wake-word need a
 * whisper-cli binary on the server host (Phase 2) and currently throw the same
 * WHISPER_NOT_INSTALLED / WHISPER_MODEL_MISSING errors the renderer already
 * handles.
 */

import { exec } from 'child_process';
import { VadProcessor, VadEvent } from './vad.js';
import { transcribeCloud, CloudProvider } from './cloud-whisper.js';
import { classifyCommand } from './command-classifier.js';
import {
  VoiceCommand,
  BuiltinOverride,
  ShellAction,
  getAllCommands,
  getBuiltinDefaults,
  matchCommand,
} from './commands.js';
import { loadSttConfig, saveSttConfig } from './config-store.js';

type MicMode = 'off' | 'global' | 'push-to-talk' | 'wake-word';
type SttBackend = 'local' | 'openai' | 'groq';

export type SttEmit = (channel: string, data?: unknown) => void;

interface SmartMatch {
  command: VoiceCommand;
  param: string;
  method: 'gpt5' | 'regex';
}

export class SttSession {
  private mode: MicMode = 'off';
  private backend: SttBackend;
  private openaiApiKey: string;
  private groqApiKey: string;
  private modelSize: string;
  private language: string;
  private smartMatching: boolean;
  private silenceTimeoutMs: number;
  private maxSpeechMs: number;
  private wakePhrase: string;
  private customCommands: VoiceCommand[];
  private builtinOverrides: Record<string, BuiltinOverride>;

  private speaking = false;
  private vad: VadProcessor | null = null;
  private feedSamplesFn: ((samples: Float32Array) => void) | null = null;

  constructor(
    private emit: SttEmit,
    private opts: { isLoopback: boolean },
  ) {
    const cfg = loadSttConfig();
    // Default to Groq on web — local Whisper isn't available in the browser, and
    // Groq is the recommended cloud backend (best Vietnamese accuracy, cheapest).
    this.backend = (cfg.backend as SttBackend) || 'groq';
    this.openaiApiKey = cfg.openaiApiKey || '';
    this.groqApiKey = cfg.groqApiKey || '';
    this.modelSize = cfg.modelSize || 'small';
    this.language = cfg.language || 'en';
    this.smartMatching = cfg.smartMatching !== false;
    this.silenceTimeoutMs = cfg.silenceTimeoutMs || 800;
    this.maxSpeechMs = cfg.maxSpeechMs || 30_000;
    this.wakePhrase = cfg.wakePhrase || 'hey octoally';
    this.customCommands = cfg.customCommands || [];
    this.builtinOverrides = cfg.builtinOverrides || {};
  }

  // -------------------------------------------------------------------------
  // Audio input (binary frames from the WebSocket)
  // -------------------------------------------------------------------------

  feedSamples(samples: Float32Array) {
    if (this.feedSamplesFn) this.feedSamplesFn(samples);
  }

  dispose() {
    this.stopCapture();
  }

  // -------------------------------------------------------------------------
  // Command dispatch (mirrors the desktop ipcMain.handle set)
  // -------------------------------------------------------------------------

  async dispatch(cmd: string, args: any): Promise<unknown> {
    switch (cmd) {
      case 'stt_get_config':
        return {
          backend: this.backend,
          openaiApiKey: this.openaiApiKey,
          groqApiKey: this.groqApiKey,
          modelSize: this.modelSize,
          language: this.language,
          wakePhrase: this.wakePhrase,
          smartMatching: this.smartMatching,
          silenceTimeoutMs: this.silenceTimeoutMs,
          maxSpeechMs: this.maxSpeechMs,
          // The browser always captures the mic via Web Audio and streams PCM.
          rendererCapture: true,
          // Local Whisper / wake-word aren't available on web (Phase 1).
          localWhisper: false,
        };

      case 'stt_start':
        return this.start(args?.mode);

      case 'stt_stop':
      case 'stt_unload_model':
        this.stopCapture();
        return;

      case 'stt_push_audio': {
        const s = args?.samples;
        if (s) this.feedSamples(s instanceof Float32Array ? s : Float32Array.from(s));
        return;
      }

      case 'stt_set_backend':
        return this.setBackend(args);

      case 'stt_set_language': {
        const lang = (args?.language || '').trim();
        const allowed = ['auto', 'en', 'vi'];
        if (!allowed.includes(lang)) {
          throw new Error(`Unknown language: ${lang}. Use ${allowed.join(', ')}.`);
        }
        this.language = lang;
        saveSttConfig({ language: lang });
        return;
      }

      case 'stt_set_wake_phrase': {
        const phrase = (args?.wakePhrase || '').trim();
        if (!phrase) throw new Error('Wake phrase cannot be empty.');
        this.wakePhrase = phrase;
        saveSttConfig({ wakePhrase: phrase });
        return;
      }

      case 'stt_set_silence_timeout': {
        const ms = Math.max(200, Math.min(5000, args?.silenceTimeoutMs));
        this.silenceTimeoutMs = ms;
        this.vad?.setSilenceTimeout(ms);
        saveSttConfig({ silenceTimeoutMs: ms });
        return;
      }

      case 'stt_set_max_speech': {
        const ms = Math.max(10_000, Math.min(300_000, args?.maxSpeechMs));
        this.maxSpeechMs = ms;
        this.vad?.setMaxSpeechMs(ms);
        saveSttConfig({ maxSpeechMs: ms });
        return;
      }

      case 'stt_set_smart_matching': {
        this.smartMatching = !!args?.enabled;
        if (args?.openaiApiKey !== undefined) this.openaiApiKey = args.openaiApiKey;
        saveSttConfig({ smartMatching: this.smartMatching, openaiApiKey: this.openaiApiKey });
        return;
      }

      case 'stt_get_voice_commands':
        return {
          commands: getAllCommands(this.customCommands, this.builtinOverrides),
          builtinDefaults: getBuiltinDefaults(),
        };

      case 'stt_set_voice_commands':
        this.customCommands = args?.customCommands || [];
        this.builtinOverrides = args?.builtinOverrides || {};
        saveSttConfig({
          customCommands: this.customCommands,
          builtinOverrides: this.builtinOverrides,
        });
        return;

      case 'stt_status':
        return {
          mode: this.mode,
          backend: this.backend,
          modelLoaded:
            this.backend === 'openai' ? !!this.openaiApiKey
              : this.backend === 'groq' ? !!this.groqApiKey
                : false,
          modelSize: this.modelSize,
          speaking: this.speaking,
          wakeWordPhase: null,
        };

      case 'stt_list_devices':
        // The browser always uses its default input device via getUserMedia.
        return [{
          name: 'default',
          displayName: 'System Default',
          description: 'Browser default microphone',
          isDefault: true,
          isHardware: true,
          formats: [],
        }];

      case 'stt_set_device':
        return; // no-op — getUserMedia uses the OS default

      case 'stt_check_model':
        return { installed: false, modelSize: this.modelSize, path: '', sizeBytes: null, active: true };

      case 'stt_list_models':
        return ['tiny', 'small', 'medium'].map((size) => ({
          installed: false,
          modelSize: size,
          path: '',
          sizeBytes: null,
          active: size === this.modelSize,
        }));

      case 'stt_check_whisper':
        return { installed: false, path: null };

      case 'stt_enter_command_mode':
        return; // wake-word only (Phase 2)

      case 'stt_set_model':
      case 'stt_download_model':
      case 'stt_install_whisper':
        throw new Error('Local Whisper is not available in the browser. Use the Groq or OpenAI backend in Speech settings.');

      default:
        throw new Error(`Unknown STT command: ${cmd}`);
    }
  }

  // -------------------------------------------------------------------------
  // Start / stop
  // -------------------------------------------------------------------------

  private async start(mode: string) {
    if (mode !== 'global' && mode !== 'push-to-talk' && mode !== 'wake-word') {
      throw new Error(`Unknown mode: ${mode}. Use 'global', 'push-to-talk', or 'wake-word'.`);
    }

    if (mode === 'wake-word') {
      throw new Error(
        'WHISPER_MODEL_MISSING: Wake word mode needs local Whisper, which is not available in the browser. ' +
        'Use cloud dictation (Groq or OpenAI) instead.',
      );
    }

    if (this.backend === 'local') {
      throw new Error(
        'WHISPER_NOT_INSTALLED: Local Whisper is not available in the browser. ' +
        'Choose the Groq or OpenAI backend in Speech settings.',
      );
    }

    // Cloud backend
    const provider = this.backend as CloudProvider;
    const apiKey = this.backend === 'groq' ? this.groqApiKey : this.openaiApiKey;
    if (!apiKey) {
      throw new Error(`${this.backend === 'groq' ? 'Groq' : 'OpenAI'} API key not set. Configure it in Speech settings.`);
    }

    if (!this.feedSamplesFn) {
      const vad = new VadProcessor(16000, this.silenceTimeoutMs, this.maxSpeechMs);
      this.startAudioSource(vad, (event) => this.handleVadEventCloud(event, provider, apiKey));
    }

    this.mode = mode as MicMode;
  }

  private setBackend(args: { backend?: string; openaiApiKey?: string; groqApiKey?: string }) {
    const backend = args?.backend as SttBackend;
    if (backend !== 'local' && backend !== 'openai' && backend !== 'groq') {
      throw new Error(`Unknown backend: ${backend}. Use 'local', 'openai', or 'groq'.`);
    }
    if (backend === 'openai' && !args.openaiApiKey && !this.openaiApiKey) {
      throw new Error('OpenAI API key is required for cloud transcription.');
    }
    if (backend === 'groq' && !args.groqApiKey && !this.groqApiKey) {
      throw new Error('Groq API key is required for Groq transcription.');
    }

    if (this.mode !== 'off' && backend !== this.backend) {
      this.stopCapture();
    }

    this.backend = backend;
    if (args.openaiApiKey !== undefined) this.openaiApiKey = args.openaiApiKey;
    if (args.groqApiKey !== undefined) this.groqApiKey = args.groqApiKey;

    saveSttConfig({ backend, openaiApiKey: this.openaiApiKey, groqApiKey: this.groqApiKey });
  }

  private startAudioSource(vad: VadProcessor, handle: (event: VadEvent) => void) {
    this.vad = vad;
    this.feedSamplesFn = (samples: Float32Array) => {
      for (const event of vad.process(samples)) handle(event);
    };
  }

  private stopCapture() {
    if (this.feedSamplesFn) {
      this.feedSamplesFn = null;
      // Tell the renderer to stop its Web Audio mic (server-initiated stops like
      // a "stop listening" voice command).
      this.emit('stt://stop-capture');
    }
    this.vad = null;
    this.mode = 'off';
    this.speaking = false;
  }

  /** Briefly mute VAD so an audio-cue beep isn't picked up as speech. */
  private muteVad() {
    this.vad?.mute(500);
  }

  // -------------------------------------------------------------------------
  // Smart command matching (GPT-5 mini with regex fallback)
  // -------------------------------------------------------------------------

  private extractTrailingNumber(text: string): string {
    const numberWords: Record<string, string> = {
      one: '1', two: '2', three: '3', four: '4', five: '5',
      six: '6', seven: '7', eight: '8', nine: '9', ten: '10',
      first: '1', second: '2', third: '3', fourth: '4', fifth: '5',
    };
    const normalized = text.toLowerCase().replace(/[^\w\s]/g, '').trim();
    const words = normalized.split(/\s+/);
    const last = words[words.length - 1];
    if (/^\d+$/.test(last)) return last;
    if (numberWords[last]) return numberWords[last];
    return '';
  }

  private extractParamFromText(text: string, cmd: VoiceCommand): string {
    const normalized = text.toLowerCase().replace(/[^\w\s]/g, '').trim();
    for (const trigger of cmd.triggerPhrases) {
      const normTrigger = trigger.toLowerCase().replace(/[^\w\s]/g, '').trim();
      if (normalized.startsWith(normTrigger)) {
        const remainder = normalized.slice(normTrigger.length).trim();
        if (remainder) return remainder;
      }
    }
    return '';
  }

  private async matchCommandSmart(text: string): Promise<SmartMatch | null> {
    const commands = getAllCommands(this.customCommands, this.builtinOverrides).filter((c) => c.enabled);

    if (this.smartMatching && this.openaiApiKey) {
      const commandInfos = commands.map((c) => ({
        id: c.id,
        name: c.name,
        actionKind: c.action.kind,
        actionTarget: 'target' in c.action ? (c.action as { target?: string }).target : undefined,
      }));

      const result = await classifyCommand(text, this.openaiApiKey, commandInfos);
      if (result) {
        const cmd = commands.find((c) => c.id === result.commandId);
        if (cmd) {
          let param = result.param;
          if (!param) {
            if (cmd.action.kind === 'navigate') {
              param = this.extractParamFromText(text, cmd);
              if (!param) param = this.extractTrailingNumber(text);
            }
          }
          const numberWords: Record<string, string> = {
            one: '1', two: '2', three: '3', four: '4', five: '5',
            six: '6', seven: '7', eight: '8', nine: '9', ten: '10',
          };
          const paramLower = param.toLowerCase().trim();
          if (numberWords[paramLower]) param = numberWords[paramLower];

          return { command: cmd, param, method: 'gpt5' };
        }
      }
    }

    const regexMatch = matchCommand(text, this.customCommands, this.builtinOverrides);
    if (regexMatch) {
      return { command: regexMatch.command, param: regexMatch.param, method: 'regex' };
    }

    return null;
  }

  // -------------------------------------------------------------------------
  // VAD event handler (cloud backend)
  // -------------------------------------------------------------------------

  private handleVadEventCloud(event: VadEvent, provider: CloudProvider, apiKey: string) {
    switch (event.type) {
      case 'utterance':
        this.muteVad();
        this.emit('stt://transcribing');
        transcribeCloud(provider, apiKey, event.samples, undefined, this.language)
          .then(async (text) => {
            if (!text) { this.muteVad(); return; }
            // global / push-to-talk run an always-on command pass before dictation
            if (this.mode === 'global' || this.mode === 'push-to-talk') {
              const matched = await this.matchCommandSmart(text);
              if (matched) {
                const cmd = matched.command;
                if (cmd.action.kind === 'stop-listening') {
                  this.muteVad();
                  this.emit('stt://voice-command', { commandId: cmd.id, action: cmd.action, param: matched.param, rawText: text });
                  this.stopCapture();
                  return;
                }
                if (cmd.action.kind === 'shell') {
                  this.runShell(cmd.action as ShellAction);
                }
                this.muteVad();
                this.emit('stt://voice-command', { commandId: cmd.id, action: cmd.action, param: matched.param, rawText: text });
              } else {
                this.muteVad();
                this.emit('stt://transcription', { text, isFinal: true });
              }
            } else {
              this.muteVad();
              this.emit('stt://transcription', { text, isFinal: true });
            }
          })
          .catch((err) => {
            console.error(`[STT] ${provider} transcription error: ${err}`);
          });
        break;

      case 'speaking-changed':
        this.speaking = event.speaking;
        this.emit('stt://vad-status', { speaking: event.speaking });
        break;

      case 'calibrated':
        this.muteVad();
        this.emit('stt://ready');
        break;
    }
  }

  /**
   * Run a custom shell voice command on the server host.
   * Only executes for loopback (localhost) connections — for a remote/LAN client
   * we refuse to run host commands and just let the event fire on the renderer.
   */
  private runShell(action: ShellAction) {
    if (!action.background) return;
    if (!this.opts.isLoopback) {
      console.error('[STT] Refusing to run shell voice command for non-loopback connection');
      return;
    }
    exec(action.command);
  }
}
