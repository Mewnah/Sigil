const WebSocket = require('ws');

const ws = new WebSocket('ws://127.0.0.1:3030/pubsub?id=test-script');

ws.on('open', function open() {
    console.log('connected');
    // Send a binary message (simulating audio)
    const array = new Float32Array(1024);
    ws.send(array.buffer);
    console.log('sent binary data');
    setTimeout(() => {
        ws.close();
        process.exit(0);
    }, 1000);
});

ws.on('error', function error(err) {
    console.error('WebSocket error:', err);
    process.exit(1);
});

ws.on('message', function message(data) {
    console.log('received: %s', data);
});
