import { floatSamplesToWavBlob } from "../encodeWav";
import { STT_State } from "../schema";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { isAbortError, raceWithTimeoutAndAbort } from "../abort";
import { RUST_CAPTURE_SAMPLE_RATE, RustCapturePollingService, truncateForUi } from "../rustCapturePolling";

/** Avoid hung `fetch` when the local server stops responding. */
const LOCAL_STT_FETCH_TIMEOUT_MS = 120_000;

/** Reads the whole body as UTF-8 through `reader`, so the caller keeps the handle it needs to cancel the read. */
async function readBodyText(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<string> {
  const decoder = new TextDecoder();
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    text += decoder.decode(value, { stream: true });
  }
  return text + decoder.decode();
}

function transcriptionsUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return `${trimmed}/audio/transcriptions`;
}

export class STT_OpenAI_AudioService extends RustCapturePollingService {
  protected readonly logTag = "[Local HTTP STT]";
  protected readonly errorLabel = "Local STT";

  protected inputDevice(params: STT_State): string {
    return params.openai_audio.device;
  }

  protected async prepare(params: STT_State): Promise<string | null> {
    const config = params.openai_audio;
    if (!config.baseUrl.trim())
      return "Local transcription API: set a base URL (e.g. http://127.0.0.1:8000/v1).";
    if (!config.model.trim())
      return "Local transcription API: set the model id your local server expects (see server docs).";
    this.receiver.onInterim("Connecting to local transcription server...");
    return null;
  }

  protected async transcribe(samples: Float32Array, signal: AbortSignal): Promise<string> {
    const config = window.ApiServer.state.services.stt.data.openai_audio;
    const base = config.baseUrl.trim();
    const model = config.model.trim();
    if (!base || !model)
      throw new Error("base URL and model are required.");

    const form = new FormData();
    form.append("file", floatSamplesToWavBlob(samples, RUST_CAPTURE_SAMPLE_RATE), "audio.wav");
    form.append("model", model);
    const lang = config.language?.trim();
    if (lang) {
      form.append("language", lang);
    }

    const headers: HeadersInit = {};
    if (config.apiKey.trim()) {
      headers.Authorization = `Bearer ${config.apiKey.trim()}`;
    }

    let res: Response;
    try {
      res = await fetchWithTimeout(transcriptionsUrl(base), {
        method: "POST",
        body: form,
        headers,
        signal,
        timeoutMs: LOCAL_STT_FETCH_TIMEOUT_MS,
      });
    } catch (error) {
      // `signal` aborts only on teardown; any other abort is the timeout.
      if (isAbortError(error) && !signal.aborted)
        throw new Error(`request timed out after ${LOCAL_STT_FETCH_TIMEOUT_MS / 1000}s`);
      throw error;
    }

    // fetchWithTimeout stops watching `signal` and its timer once headers arrive, so bound the body read here:
    // a server that stalls mid-body must not pin the pipeline past stop() or forever. On abort or timeout the
    // body is cancelled too, so the connection and the pending read are released instead of left running.
    const reader = res.body?.getReader();
    let raw: string;
    try {
      raw = await raceWithTimeoutAndAbort(
        reader ? readBodyText(reader) : res.text(),
        LOCAL_STT_FETCH_TIMEOUT_MS,
        "response body timed out",
        signal,
      );
    } catch (error) {
      void reader?.cancel().catch(() => {});
      throw error;
    }
    if (!res.ok) {
      const hint = truncateForUi(raw, 120);
      throw new Error(hint ? `HTTP ${res.status}: ${hint}` : `HTTP ${res.status}`);
    }
    let json: { text?: string; error?: { message?: string } };
    try {
      json = (JSON.parse(raw) ?? {}) as typeof json;
    } catch {
      throw new Error("server returned non-JSON response.");
    }
    if (json.error?.message)
      throw new Error(json.error.message);
    return json.text ?? "";
  }
}
