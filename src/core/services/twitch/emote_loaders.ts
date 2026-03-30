import { fetchWithTimeout } from "@/utils/fetchWithTimeout";

const EMOTE_FETCH_TIMEOUT_MS = 25_000;

//region BTTV
export async function Load_BTTV_GLOBAL() {
  const emotes: Record<string, string>                     = {};
  try {
    const bttv_global                          = await fetchWithTimeout('https://api.betterttv.net/3/cached/emotes/global', { timeoutMs: EMOTE_FETCH_TIMEOUT_MS });
    const json_bttv_global: any = await bttv_global.json();
    for (let i = 0; i < json_bttv_global.length; i++)
      emotes[json_bttv_global[i].code] = `https://cdn.betterttv.net/emote/${json_bttv_global[i].id}/1x`
  } catch (error) {}
  return emotes;
}

export async function Load_BTTV_CHANNEL(id: string) {
  const emotes: Record<string, string>                       = {};
  try {
    const bttv_channel                           = await fetchWithTimeout(`https://api.betterttv.net/3/cached/users/twitch/${id}`, { timeoutMs: EMOTE_FETCH_TIMEOUT_MS });
    const json_bttv_channel: any = await bttv_channel.json();
    for (let i = 0; i < json_bttv_channel.channelEmotes.length; i++)
      emotes[json_bttv_channel.channelEmotes[i].code] = `https://cdn.betterttv.net/emote/${json_bttv_channel.channelEmotes[i].id}/1x`
    for (let i = 0; i < json_bttv_channel.sharedEmotes.length; i++)
      emotes[json_bttv_channel.sharedEmotes[i].code] = `https://cdn.betterttv.net/emote/${json_bttv_channel.sharedEmotes[i].id}/1x`
  } catch (error) {}
  return emotes;
}
//endregion

//region FFZ
function ParseFFz(data: unknown) {
  const emotes: Record<string, string> = {};
  try {
    const o = data as { sets?: Record<string, { emoticons: { name: string; urls: Record<string, string> }[] }> };
    const sets = o?.sets;
    if (!sets) return emotes;
    Object.keys(sets).forEach(set_key => {
      for (let i = 0; i < sets[set_key].emoticons.length; i++) {
        const emoticon = sets[set_key].emoticons[i];
        emotes[emoticon.name] = emoticon.urls["1"]
      }
    });
  } catch (error) {}
  return emotes;
}
export async function Load_FFZ_GLOBAL() {
  try {
    const res = await fetchWithTimeout("https://api.frankerfacez.com/v1/set/global", { timeoutMs: EMOTE_FETCH_TIMEOUT_MS });
    if (!res.ok) return {};
    const json_ffz_global: unknown = await res.json();
    return ParseFFz(json_ffz_global);
  } catch {
    return {};
  }
}

/** Maps `/v1/rooms/{login}` bulk response (always 200; empty emotes if no FFZ room). */
function ParseFFzRoomsLimited(data: unknown) {
  const emotes: Record<string, string> = {};
  try {
    const o = data as {
      template?: { static?: string };
      emotes?: { id: number; name: string }[];
    };
    const tpl = o.template?.static;
    const list = o.emotes;
    if (!tpl || !list?.length) return emotes;
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      emotes[e.name] = tpl.replace("{id}", String(e.id)).replace("{scale}", "1");
    }
  } catch {
    //
  }
  return emotes;
}

/** Twitch login (lowercase). Uses `/v1/rooms/{login}` so channels without an FFZ room get 200 + empty emotes (no 404). */
export async function Load_FFZ_CHANNEL(channelLogin: string) {
  const login = channelLogin?.trim().toLowerCase();
  if (!login) return {};
  try {
    const res = await fetchWithTimeout(
      `https://api.frankerfacez.com/v1/rooms/${encodeURIComponent(login)}`,
      { timeoutMs: EMOTE_FETCH_TIMEOUT_MS }
    );
    if (!res.ok) return {};
    const json: unknown = await res.json();
    return ParseFFzRoomsLimited(json);
  } catch {
    return {};
  }
}
//endregion

//region 7tv
export async function Load_7TV_CHANNEL(id: string) {
  
  try {

    const resp                          = await fetchWithTimeout(`https://7tv.io/v3/users/twitch/${id}`, { timeoutMs: EMOTE_FETCH_TIMEOUT_MS });
    const r = await resp.json();
    const emoteSet = r.emote_set;
    return Object.fromEntries(emoteSet.emotes.map((emote: any) =>
      [emote.name, `${emote.data.host.url}/${emote.data.host.files[1].name}`]
    ))
  } catch (error) {
    return {}
  }
}
export async function Load_7TV_GLOBAL() {
  try {
    const resp                          = await fetchWithTimeout(`https://7tv.io/v3/emote-sets/global`, { timeoutMs: EMOTE_FETCH_TIMEOUT_MS });
    const r = await resp.json();
    return Object.fromEntries(r.emotes.map((emote: any) =>
      [emote.name, `${emote.data.host.url}/${emote.data.host.files[1].name}`]
    ));
  } catch (error) {
    return {}
  }
}
//endregion
