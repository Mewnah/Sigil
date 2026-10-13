/**
 * The Speech to Text card of the footer while the microphone delivers only digital silence: "NO SIGNAL" in warning
 * colors instead of "ONLINE", the full sentence ("Sigil hears nothing from <mic>. ...") as its tooltip, and nothing
 * of it for a service that is stopped or hears its microphone. Real English strings plus the staged ones.
 */
import { renderToStaticMarkup } from "react-dom/server";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { info: vi.fn() }) }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { ServiceNetworkState } from "@/types";
import { StatusCard, sttCardNotice } from "../StatusFooter";

beforeAll(async () => {
  i18n.use(initReactI18next);
  await initSttTestI18n();
});

const t = (key: string, options?: Record<string, unknown>) => i18n.t(key, options) as string;
const decode = (markup: string) => markup.replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&amp;/g, "&");

describe("sttCardNotice", () => {
  it("is the short word and the full sentence while the service runs and the microphone is silent", () => {
    const notice = sttCardNotice({ status: ServiceNetworkState.connected, silentInput: { device: "Headset Mic" } }, t);
    expect(notice?.label).toBe("No signal");
    expect(notice?.title).toContain("Sigil hears nothing from Headset Mic.");
    expect(notice?.title).toContain("Let desktop apps access your microphone");
  });

  it("is nothing for a microphone that is heard, and for a service that is not running", () => {
    expect(sttCardNotice({ status: ServiceNetworkState.connected, silentInput: null }, t)).toBeUndefined();
    expect(sttCardNotice({ status: ServiceNetworkState.connecting, silentInput: { device: "Headset Mic" } }, t)).toBeUndefined();
    expect(sttCardNotice({ status: ServiceNetworkState.disconnected, silentInput: { device: "Headset Mic" } }, t)).toBeUndefined();
  });
});

describe("StatusCard", () => {
  const card = (props: { status?: ServiceNetworkState; notice?: { label: string; title: string } }) =>
    decode(
      renderToStaticMarkup(
        <StatusCard label="Speech to Text" icon={<i />} status={props.status ?? ServiceNetworkState.connected} serviceId="stt" notice={props.notice} />,
      ),
    );

  it("shows the notice instead of ONLINE, in warning colors, with the sentence as its tooltip", () => {
    const out = card({ notice: sttCardNotice({ status: ServiceNetworkState.connected, silentInput: { device: "Headset Mic" } }, t) });
    expect(out).toContain(">No signal<");
    expect(out).not.toContain("ONLINE");
    expect(out).toMatch(/class="[^"]*text-warning[^"]*">No signal</);
    expect(out).toContain('title="Sigil hears nothing from Headset Mic. Check Windows microphone privacy (Settings > Privacy & security > Microphone > Let desktop apps access your microphone)');
    expect(out).toContain("bg-warning/10 text-warning");
    expect(out).not.toContain("bg-success/10");
  });

  it("is unchanged without a notice", () => {
    const online = card({});
    expect(online).toContain(">ONLINE<");
    expect(online).toContain("bg-success/10 text-success");
    expect(online).not.toContain("title=\"Sigil");
    expect(card({ status: ServiceNetworkState.disconnected })).toContain(">OFFLINE<");
    expect(card({ status: ServiceNetworkState.connecting })).toContain(">CONNECTING...<");
  });
});
