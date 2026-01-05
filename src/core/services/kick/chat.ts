export default class KickChatApi {
    constructor() { }

    connect(channel: string) {
        console.log("KickChatApi: connect", channel);
    }

    disconnect() {
        console.log("KickChatApi: disconnect");
    }

    dispose() {
        console.log("KickChatApi: dispose");
    }

    post(message: string) {
        console.log("KickChatApi: post", message);
    }
}
