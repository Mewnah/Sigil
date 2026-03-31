import { devLog } from "@/utils/devLog";

export default class KickEmotesApi {
    constructor() { }

    dictionary: Record<string, string> = {};

    loadEmotes(channelId: string) {
        devLog("KickEmotesApi: loadEmotes", channelId);
    }

    dispose() {
        devLog("KickEmotesApi: dispose");
    }
}
