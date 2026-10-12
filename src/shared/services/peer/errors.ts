import i18n from "i18next";

/** Shown on a `/client` mirror the host refused because of the LAN token (P6-D). */
export function invalidKeyMessage() {
  return i18n.t("client.peer_invalid_key");
}

/**
 * `startClient` rejects with this when the host refused the mirror's LAN token (P6-D): no retry can
 * fix it, so the page shows the refusal. Any other connection failure is retried inside the page.
 */
export class MirrorRefusedError extends Error {
  constructor() {
    super(invalidKeyMessage());
    this.name = "MirrorRefusedError";
  }
}

/**
 * `startClient` rejects with this for the start-up failures that no retry can fix and that are not
 * about the LAN token: a browser without WebRTC, a Peer that cannot even be constructed (a bad
 * host or port). Connection failures never end up here; they are retried inside the page.
 */
export class MirrorStartError extends Error {
  /**
   * `secret` (the LAN token) is cut out of the message: the browser's error for a bad WebSocket URL
   * quotes the whole URL, and this message is shown on the page.
   */
  constructor(readonly original: unknown, secret = "") {
    const text = original instanceof Error && original.message ? original.message : String(original ?? "the mirror could not start");
    super(secret ? text.split(secret).join("…") : text);
    this.name = "MirrorStartError";
  }
}
