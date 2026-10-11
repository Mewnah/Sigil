import { TextEventType, type TextEvent } from "@/types";
import { isCancelMarker } from "./translation/sequencer";

/**
 * The text of a final that may be posted to a public destination (Twitch / Kick chat, a Discord webhook), or `null`.
 *
 * STT closes a sentence that is cut off (a mute, VRChat's mute sync, a stop, a new session) with the `[...]` cancel
 * marker as a final. It is caption punctuation for the consumers that show the open sentence (captions, history, the
 * chatbox's typing bubble): posting it would put a bare "[...]" in the stream chat every time the user mutes
 * mid-sentence.
 */
export function postableFinal(data?: TextEvent): string | null {
  if (!data || data.type !== TextEventType.final || !data.value || isCancelMarker(data.value)) return null;
  return data.value;
}
