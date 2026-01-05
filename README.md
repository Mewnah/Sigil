# Sigil

**A comprehensive voice communication suite for VRChat and streaming.**

Sigil is a powerful, open-source application for real-time speech-to-text, text-to-speech, translation, and stream captioning. Built for the VRChat community and streamers who need accessible, customizable voice tools.

![Demo](https://user-images.githubusercontent.com/3977499/218335391-a53dab5b-1e22-47b8-89c5-e1124798fbdc.gif)

## ✨ Features

### Speech Recognition (STT)

- **Local Whisper** - Privacy-focused, runs entirely on your machine
- **Microsoft Azure** - Cloud-based with high accuracy
- **Deepgram** - Real-time streaming transcription
- **WebSpeech API** - Browser-based (Chrome/Edge)
- *Coming Soon: Vosk integration*

### Text-to-Speech (TTS)

- **Microsoft Azure** - Natural-sounding voices
- **Windows SAPI** - Built-in system voices
- **TikTok Voices** - Trending voice styles
- **WebSpeech API** - Browser-based synthesis
- *Coming Soon: VoiceVox integration*

### AI Transform

- Rewrite your speech in real-time using AI
- Support for **OpenAI**, **OpenRouter**, and **Local LLMs**
- Display original and transformed text side-by-side

### Streaming Integration

- **OBS Studio** - Native browser source captions
- **Twitch** - Post to chat, use 7TV/FFZ/BTTV emotes
- **Kick** - Full OAuth integration with chat support
- **Discord** - Send transcriptions to channels

### VRChat

- **KillFrenzy Avatar Text** support
- Native VRChat chatbox integration
- Real-time subtitle display

### Customization

- Canvas editor with drag & drop
- Custom fonts, colors, shadows, backgrounds
- Text animations and particle effects
- Multiple scenes with auto-switching

## 🚀 Getting Started

### Quick Setup with OBS

1. **Open Sigil** and copy the browser source link
2. **Create a Browser Source** in OBS
3. Paste the link and set dimensions to match canvas (default: 500x300)

*Or click "Set Up OBS" for automatic configuration via obs-websocket.*

![OBS Setup](https://user-images.githubusercontent.com/3977499/218330675-472e02a9-1e18-4d60-8662-c4ca33325c24.gif)

## 🗺️ Roadmap

- [ ] Whisper FFI backend for stable local inference
- [ ] Vosk STT integration
- [ ] VoiceVox TTS integration
- [ ] Lightweight voice changer
- [ ] ASL-to-text (experimental)

## 🛠️ Development

```bash
# Install dependencies
pnpm install

# Run in development mode
pnpm tauri dev

# Build for production
pnpm tauri build
```

### Environment Variables

Create a `.env` file with your OAuth credentials:

```env
# Twitch (https://dev.twitch.tv/console/apps)
SIGIL_TWITCH_CLIENT_ID=your_client_id

# Kick (https://kick.com/settings/developer)
SIGIL_KICK_CLIENT_ID=your_client_id
SIGIL_KICK_CLIENT_SECRET=your_client_secret
```

---

## 🙏 Credits & Acknowledgments

**Sigil is built upon the foundation of [Curses](https://github.com/mmpneo/curses) by mmpneo.**

This project would not exist without the incredible work of **mmpneo** and everyone who contributed to and supported the original Curses application. Sigil continues their vision of making communication accessible to everyone.

### Special Thanks

- **mmpneo** - Original creator of Curses
- **The Curses community** - For feedback, bug reports, and support
- **VRChat community** - For inspiring accessible communication tools
- All open-source contributors who made this possible

**If you enjoy Sigil, please consider starring the original [Curses repository](https://github.com/mmpneo/curses) to show your appreciation.**

---

Made with 💜 for the VRChat community
