export default class KickEmotesApi {
    constructor() { }

    dictionary: Record<string, string> = {};

    loadEmotes(channelId: string) {
        console.log("KickEmotesApi: loadEmotes", channelId);
    }

    dispose() {
        console.log("KickEmotesApi: dispose");
    }
}
