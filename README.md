# About Curses+

This is a modified version of the original Curses application by **mmpneo**. Maintained by **Mewnah**.

**Curses+ (v0.2.0)** includes:

- **UI/UX Refinements**: Reorganized Sidebar and cleaner interface elements.
- **Bug Fixes**: Addressed various styling and functional issues from the original codebase.
- **Improved Language Support**: Better default language handling (English default) and corrected localization keys.

<p align="center">
<!-- static -->
  <!-- <img width="600" src="https://user-images.githubusercontent.com/3977499/218319590-296c96f0-7daa-4130-ab40-6b32f20cc26e.png"> -->
  <img width="600" src="https://user-images.githubusercontent.com/3977499/218335391-a53dab5b-1e22-47b8-89c5-e1124798fbdc.gif" alt="Curses Demo GIF">
</p>

# Features

- **Native OBS stream captions**
- **OBS Captions customization**: Colors, fonts, shadows, background textures, text typing animation, sound effects, particle effects and CSS
- **AI Transform**: Rewrite your voice in real-time using OpenAI, OpenRouter, or Local LLMs
- **Synchronized Subtitles**: Display original text alongside AI-transformed text with perfect timing
- **Canvas Editor**: Drag & Drop interface with Snap-to-Grid and smart element alignment
- **Speech to Text**: [Microsoft Azure](https://azure.microsoft.com/en-au/products/cognitive-services/speech-to-text/), [Deepgram](https://deepgram.com/), WebSpeechApi (Chrome/Edge), Local OpenAI Whisper (beta)
- **Text to Speech**: [Microsoft Azure](https://azure.microsoft.com/en-us/products/cognitive-services/text-to-speech/), [Uberduck](https://uberduck.ai/), TikTok, Windows API (SAPI), WebSpeechAPI
- **VRChat**: [KillFrenzy Avatar text](https://github.com/killfrenzy96/KillFrenzyAvatarText), VRChat's chatbox
- **Twitch**:
  - Use 7TV/FFZ/BTTV emotes in OBS captions
  - Post your STT to chat
  - Use your chat messages as a source for captions and TTS
  - native captions
- **Discord**: Send your STT to specified channel
- **Scenes**:
  - Save multiple designs and freely switch between them
  - Automatically switch design when OBS changes scene

# Getting Started with OBS

### 1. Open app and copy link for OBS

Or click "Set Up OBS" to have everything set up automatically with **obs-websocket** plugin

<img width="600" src="https://user-images.githubusercontent.com/3977499/218330675-472e02a9-1e18-4d60-8662-c4ca33325c24.gif" alt="Obs Setup GIF">

### 2. Create Browser Source in OBS

Paste the link and change window size to match app's canvas size (default is 500x300)

<img width="600" src="https://user-images.githubusercontent.com/3977499/218331723-721b69c5-a457-4dad-9658-f5232afc68f1.gif" alt="Browser Source Setup GIF">

## Roadmap

- [ ] **Whisper Refactor**: Rewrite backend to use FFI for stable local inference
- [ ] **STT**: Vosk Integration
- [ ] **TTS**: VoiceVox Integration



**Special thanks and all credits go to mmpneo and everyone who has contributed to and supported the original Curses application.**

**Without them, none of this would be possible.**
