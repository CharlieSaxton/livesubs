# Livesubs

Paste a video link, get live translated subtitles. Everything runs in your browser —
no server, no account, no API key, no cost.

**Live: https://charliesaxton.github.io/livesubs/**

## How it works

1. Paste a link (YouTube, or a direct `.mp4` / `.m3u8`). The video embeds in the page.
2. Press **Start captions** and share the tab **with audio** when Chrome asks.
3. [Whisper](https://huggingface.co/onnx-community/whisper-base) transcribes the audio on your
   GPU via WebGPU; a [Helsinki Opus-MT](https://huggingface.co/Xenova/opus-mt-fr-en) model
   translates each line into English.

Model weights download once (~90 MB) from the Hugging Face CDN straight to your browser and are
cached by the browser afterwards. Audio never leaves your machine.

## Alignment

Live captions run roughly 8 seconds behind — Whisper needs a window of audio before it can
transcribe it. As you watch, the app builds a **timestamped transcript**, so rewinding or
scrubbing re-syncs the subtitles exactly. Export as `.srt` when you're done.

## Requirements

- Chrome or Edge on desktop (tab audio capture isn't available in Safari or Firefox)
- WebGPU for real-time speed; falls back to CPU, which is slower than realtime

## Languages

French, Spanish, German, Italian, Russian, Chinese, Arabic and Dutch use dedicated translation
models. Everything else falls back to a multilingual model, and if that fails, to Whisper's own
translation task.
