// Runs Whisper (speech -> text) and Opus-MT (text -> English) entirely in the browser.
// Nothing is uploaded; model weights are fetched once from the HF CDN and cached.

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

env.allowLocalModels = false;

// Dedicated one-to-one translators beat Whisper's own translate task by a wide margin
// and are only ~40 MB. Anything not listed falls back to the multilingual model.
const MT = {
  fr:'Xenova/opus-mt-fr-en', es:'Xenova/opus-mt-es-en', de:'Xenova/opus-mt-de-en',
  it:'Xenova/opus-mt-it-en', ru:'Xenova/opus-mt-ru-en', zh:'Xenova/opus-mt-zh-en',
  ar:'Xenova/opus-mt-ar-en', nl:'Xenova/opus-mt-nl-en',
};
const MT_FALLBACK = 'Xenova/opus-mt-mul-en';

// Whisper reliably invents these over silence or music. Drop them.
const GHOSTS = [
  'sous-titres réalisés', 'sous-titrage', 'amara.org', 'subtitles by',
  'thanks for watching', 'merci d\'avoir regardé', 'abonnez-vous',
  'subscribe', 'transcription', 'untertitel', 'sottotitoli',
];

let asr = null, asrId = null;
let mt = null, mtId = null;
let mtBroken = false;

const post = (m) => self.postMessage(m);

function progress(scope){
  return (p) => {
    if (p.status === 'progress' && p.total) {
      post({ type:'progress', scope, file:p.file, pct: Math.round(p.loaded / p.total * 100) });
    } else if (p.status === 'ready' || p.status === 'done') {
      post({ type:'progress', scope, pct:100 });
    }
  };
}

async function getASR(id){
  if (asr && asrId === id) return asr;
  post({ type:'status', text:`Loading ${id.split('/').pop()}…` });
  asr = await pipeline('automatic-speech-recognition', id, {
    device: self.__gpu ? 'webgpu' : 'wasm',
    dtype: self.__gpu ? { encoder_model:'fp32', decoder_model_merged:'q4' } : 'q8',
    progress_callback: progress('asr'),
  });
  asrId = id;
  return asr;
}

async function getMT(lang){
  if (mtBroken) return null;
  const id = MT[lang] || MT_FALLBACK;   // 'auto' lands on the multilingual model
  if (mt && mtId === id) return mt;
  post({ type:'status', text:'Loading translator…' });
  try {
    mt = await pipeline('translation', id, { dtype:'q8', progress_callback: progress('mt') });
    mtId = id;
    return mt;
  } catch (e) {
    if (id !== MT_FALLBACK) {                     // dedicated pair missing -> try multilingual
      try {
        mt = await pipeline('translation', MT_FALLBACK, { dtype:'q8', progress_callback: progress('mt') });
        mtId = MT_FALLBACK;
        return mt;
      } catch (_) { /* fall through */ }
    }
    mtBroken = true;                              // give up on MT; caller uses Whisper translate
    post({ type:'status', text:'Translator unavailable — using Whisper translation.' });
    return null;
  }
}

const isGhost = (s) => {
  const t = s.toLowerCase();
  return GHOSTS.some(g => t.includes(g));
};

function clean(s){
  let t = (s || '').replace(/\s+/g, ' ').trim();
  // Whisper loops on unclear audio: "oui oui oui oui oui". Collapse runs of 3+.
  t = t.replace(/\b(\S{1,24}?)(?:[ ,]+\1\b){2,}/gi, '$1');
  return t;
}

self.onmessage = async (e) => {
  const msg = e.data;

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

  if (msg.type === 'audio'){
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
        : [{ timestamp:[0, audio.length/16000], text: out.text }];

      const cues = [];
      for (const c of raw){
        const text = clean(c.text);
        if (!text || text.length < 2 || isGhost(text)) continue;
        const [s, e2] = c.timestamp || [0, null];
        cues.push({
          start: t0 + (s ?? 0),
          end:   t0 + (e2 ?? (s ?? 0) + 2),
          src:   text,
          en:    null,
        });
      }

      if (translate && cues.length){
        const engine = await getMT(lang);
        if (engine){
          for (const cue of cues){
            try {
              const r = await engine(cue.src);
              cue.en = clean(Array.isArray(r) ? r[0]?.translation_text : r?.translation_text);
            } catch { cue.en = null; }
          }
        } else {
          // Fallback: ask Whisper itself to translate the same window.
          try {
            const tr = await rec(audio, { language: lang === 'auto' ? null : lang, task:'translate' });
            const t = clean(tr.text);
            if (t && !isGhost(t)) cues[0].en = t;
          } catch { /* leave untranslated */ }
        }
      }

      post({ type:'cues', id, cues });
    } catch (err) {
      post({ type:'cues', id, cues:[], error:String(err?.message || err) });
    }
  }
};
