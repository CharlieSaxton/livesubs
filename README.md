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

Model weights are served from this site itself, not from huggingface.co. HF's CDN delivers under
10 KB/s to some regions (New Zealand included), which would make a first run take hours; a GitHub
Actions job mirrors the weights at build time so the browser pulls them at CDN speed instead.
First run downloads ~190 MB (Whisper base + the French translator) and the browser caches it.
Audio never leaves your machine.

## Alignment

Live captions run roughly 8 seconds behind — Whisper needs a window of audio before it can
transcribe it. As you watch, the app builds a **timestamped transcript**, so rewinding or
scrubbing re-syncs the subtitles exactly. Export as `.srt` when you're done.

## Requirements

- Chrome or Edge on desktop (tab audio capture isn't available in Safari or Firefox)
- WebGPU for real-time speed; falls back to CPU, which is slower than realtime

## Languages

French uses a dedicated Opus-MT model, which is markedly better than Whisper's own translation and
is mirrored for speed. Every other language is translated by Whisper itself, which needs no extra
download. Adding another dedicated pair is one line in `worker.js` plus one in the workflow.
