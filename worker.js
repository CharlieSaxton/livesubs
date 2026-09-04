// Speech -> text -> English, entirely in the browser. Nothing is uploaded.
//
// Weights come from this site's own /models/ mirror, not huggingface.co: HF's CDN serves
// some regions at under 10 KB/s, which would make a first run take hours. Anything not
// mirrored still falls back to HF.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels  = true;
env.allowRemoteModels = true;                                   // fallback for anything unmirrored
env.localModelPath    = new URL('./models/', location.href).href;

// Only French is mirrored, and a dedicated pair model beats Whisper's own translate task
// by a wide margin. Every other language uses Whisper translation, which needs no download.
const MT = { fr: 'Xenova/opus-mt-fr-en' };

// Whisper invents these over silence and music.
const GHOSTS = [
  'sous-titres réalisés', 'sous-titrage', 'amara.org', 'subtitles by', 'subtitled by',
  'thanks for watching', "merci d'avoir regardé", 'abonnez-vous', 'subscribe',
  'untertitel', 'sottotitoli', 'подписывайтесь',
];

let asr = null, asrId = null, asrDev = null;
let mt = null, mtId = null, mtDead = false;

const post = (m) => self.postMessage(m);

const progress = (scope) => (p) => {
  if (p.status === 'progress' && p.total)
    post({ type:'progress', scope, pct: Math.round(p.loaded / p.total * 100), file: p.file });
  else if (p.status === 'ready' || p.status === 'done')
    post({ type:'progress', scope, pct: 100 });
};

async function getASR(id){
  const want = self.__gpu ? 'webgpu' : 'wasm';
  if (asr && asrId === id && asrDev === want) return asr;
  post({ type:'status', text:`Loading ${id.split('/').pop()}…` });
  try {
    asr = await pipeline('automatic-speech-recognition', id,
      { device: want, dtype:'q8', progress_callback: progress('asr') });
    asrDev = want;
  } catch (e) {
    if (want !== 'wasm'){                       // some GPUs reject int8 kernels; CPU always works
      post({ type:'status', text:'GPU path unavailable — falling back to CPU.' });
      asr = await pipeline('automatic-speech-recognition', id,
        { device:'wasm', dtype:'q8', progress_callback: progress('asr') });
      asrDev = 'wasm';
      post({ type:'device', device:'wasm' });
    } else throw e;
  }
  asrId = id;
  return asr;
}

async function getMT(lang){
  const id = MT[lang];
  if (!id || mtDead) return null;               // -> caller uses Whisper's translate task
  if (mt && mtId === id) return mt;
  post({ type:'status', text:'Loading translator…' });
  try {
    mt = await pipeline('translation', id, { dtype:'q8', progress_callback: progress('mt') });
    mtId = id;
    return mt;
  } catch {
    mtDead = true;
    return null;
  }
}

const isGhost = (s) => { const t = s.toLowerCase(); return GHOSTS.some(g => t.includes(g)); };

function clean(s){
  let t = (s || '').replace(/\s+/g, ' ').trim();
  t = t.replace(/\b(\S{1,24}?)(?:[ ,]+\1\b){2,}/gi, '$1');   // collapse Whisper's repetition loops
  return t;
}

self.onmessage = async ({ data: msg }) => {
  if (msg.type === 'init'){
    self.__gpu = msg.gpu;
    try {
      await getASR(msg.model);
      if (msg.translate) await getMT(msg.lang);
      post({ type:'ready' });
    } catch (err) {
      post({ type:'error', text: String(err?.message || err) });
    }
    return;
  }

  if (msg.type !== 'audio') return;
  const { audio, t0, id, lang, model, translate } = msg;

  try {
    const rec = await getASR(model);
    const out = await rec(audio, {
      language: lang === 'auto' ? null : lang,
      task: 'transcribe',
      return_timestamps: true,
    });

    const raw = out.chunks?.length
      ? out.chunks
      : [{ timestamp:[0, audio.length / 16000], text: out.text }];

    const cues = [];
    for (const c of raw){
      const text = clean(c.text);
      if (!text || text.length < 2 || isGhost(text)) continue;
      const [s, e] = c.timestamp || [0, null];
      cues.push({ start: t0 + (s ?? 0), end: t0 + (e ?? (s ?? 0) + 2), src: text, en: null });
    }

    if (translate && cues.length){
      const engine = await getMT(lang);
      if (engine){
        for (const cue of cues){
          try {
            const r = await engine(cue.src);
            cue.en = clean(Array.isArray(r) ? r[0]?.translation_text : r?.translation_text);
          } catch { /* leave the original */ }
        }
      } else {
        try {                                   // no dedicated pair: let Whisper translate
          const tr = await rec(audio, { language: lang === 'auto' ? null : lang, task:'translate' });
          const t = clean(tr.text);
          if (t && !isGhost(t)) cues[0].en = t;
        } catch { /* leave the original */ }
      }
    }

    post({ type:'cues', id, cues });
  } catch (err) {
    post({ type:'cues', id, cues:[], error: String(err?.message || err) });
  }
};
