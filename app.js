const SR = 16000;                 // what Whisper requires
const WIN_S = 8.0;                // seconds of audio per window
const STRIDE_S = 6.5;             // hop between windows (1.5 s of shared context)
const SILENCE = 0.0016;           // RMS floor; below this we skip the model entirely

// Safari ignores { sampleRate: 16000 } and hands back 44100 or 48000, so nothing may assume
// the rate. Audio is buffered at whatever the device gives and resampled just before the model.
let rate = SR, WIN = WIN_S * SR, STRIDE = STRIDE_S * SR;

// Box-averaged decimation. Plain interpolation aliases badly at 48k->16k, which Whisper hears
// as noise; averaging across the window is a crude but effective anti-alias filter.
function to16k(buf, from){
  if (from === SR) return buf;
  const ratio = from / SR, n = Math.floor(buf.length / ratio);
  const out = new Float32Array(n), half = Math.max(1, Math.round(ratio / 2));
  for (let i = 0; i < n; i++){
    const c = Math.round(i * ratio);
    let sum = 0, k = 0;
    for (let j = Math.max(0, c - half); j <= Math.min(buf.length - 1, c + half); j++){ sum += buf[j]; k++; }
    out[i] = sum / k;
  }
  return out;
}

const $ = (id) => document.getElementById(id);
const ui = {};
['landing','stage','urlForm','url','playerHost','capPrimary','capSecondary','startBtn','stopBtn',
 'srcLang','showMode','model','status','progressWrap','progressFill','lines','empty','engine',
 'srtBtn','txtBtn','clearBtn','reqBrowser','reqGpu','hint','source']
 .forEach(k => ui[k] = $(k));

/* ---------------------------------------------------------------- capability */
const hasGPU  = !!navigator.gpu;
const canGrab = !!navigator.mediaDevices?.getDisplayMedia;
const isChromium = !!window.chrome && !/firefox/i.test(navigator.userAgent);

ui.engine.textContent = hasGPU ? 'WebGPU' : 'CPU only';
ui.engine.className = 'pill ' + (hasGPU ? 'ok' : 'warn');
if (!hasGPU) ui.reqGpu.classList.add('bad'), ui.reqGpu.textContent = 'No WebGPU — will be slow';
const canTab = canGrab && isChromium;
if (!canTab){
  ui.reqBrowser.classList.add('bad');
  ui.reqBrowser.textContent = 'No tab audio here — use Microphone';
}

const HINTS = {
  tab: 'Chrome will ask what to share — pick <b>This tab</b> and make sure <b>Also share tab audio</b> is ticked. Nothing leaves your computer.',
  mic: 'The page listens through the microphone, so play the video out loud — on this device or another one. Keep the volume up and the room quiet. Nothing leaves your device.',
};
function syncSource(){
  ui.hint.innerHTML = HINTS[ui.source.value];
  const tabOpt = ui.source.querySelector('option[value="tab"]');
  if (tabOpt) tabOpt.disabled = !canTab;
}

/* ---------------------------------------------------------------- player */
let player = null;

function youtubeId(u){
  try{
    const url = new URL(u);
    const h = url.hostname.replace(/^www\./,'');
    if (h === 'youtu.be') return url.pathname.slice(1).split('/')[0] || null;
    if (!/(^|\.)youtube(-nocookie)?\.com$/.test(h)) return null;
    if (url.searchParams.get('v')) return url.searchParams.get('v');
    const m = url.pathname.match(/^\/(shorts|embed|live|v)\/([^/?#]+)/);
    return m ? m[2] : null;
  }catch{ return null; }
}

function ytApi(){
  if (window.YT?.Player) return Promise.resolve();
  return new Promise(res => {
    const prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => { prev?.(); res(); };
    const s = document.createElement('script');
    s.src = 'https://www.youtube.com/iframe_api';
    document.head.appendChild(s);
  });
}

async function mount(raw){
  ui.playerHost.innerHTML = '';
  const id = youtubeId(raw);

  if (id){
    await ytApi();
    const host = document.createElement('div');
    ui.playerHost.appendChild(host);
    const yt = await new Promise(res => {
      const p = new YT.Player(host, {
        videoId: id,
        playerVars: { rel:0, modestbranding:1, playsinline:1, cc_load_policy:0 },
        events: { onReady: () => res(p) },
      });
    });
    player = { time: () => yt.getCurrentTime?.() ?? 0, seek: (t) => yt.seekTo(t, true) };
    return;
  }

  if (/\.(mp4|webm|ogg|ogv|m4v|mov|mp3|m4a|wav|aac|flac)(\?|#|$)/i.test(raw) || /\.m3u8(\?|#|$)/i.test(raw)){
    const v = document.createElement('video');
    v.src = raw; v.controls = true; v.crossOrigin = 'anonymous';
    ui.playerHost.appendChild(v);
    player = { time: () => v.currentTime, seek: (t) => { v.currentTime = t; } };
    return;
  }

  const embedT0 = Date.now();
  const f = document.createElement('iframe');
  f.src = raw;
  f.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  f.referrerPolicy = 'origin';
  ui.playerHost.appendChild(f);
  player = { time: () => (Date.now() - embedT0) / 1000, seek: () => {} };
  say('Embedded site — timing is measured from load, since the page can\'t read its player.', 'warn');
}

/* ---------------------------------------------------------------- cues */
let cues = [];
let lastEnd = -1;

const fmt = (t) => {
  const s = Math.max(0, t|0);
  return `${String((s/60|0)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
};

// Windows overlap by 1.5 s, so the same words can arrive twice. Discard a cue only when it
// adds nothing new: judging by start time alone silently drops speech that straddles the seam.
const words = (s) => new Set(s.toLowerCase().replace(/[^\p{L}\p{N} ]/gu,'').split(' ').filter(Boolean));
function overlaps(a, b){
  const A = words(a), B = words(b);
  if (!A.size || !B.size) return 0;
  let hit = 0; for (const w of A) if (B.has(w)) hit++;
  return hit / Math.min(A.size, B.size);
}

function addCues(list){
  let added = false;
  for (const c of list){
    if (c.end <= lastEnd + 0.3) continue;                       // wholly inside what we have
    if (cues.slice(-2).some(p => overlaps(p.src, c.src) > 0.8)) continue;   // same words again
    if (c.start < lastEnd) c.start = lastEnd;                   // trim the seam, keep new words
    cues.push(c);
    lastEnd = Math.max(lastEnd, c.end);
    added = true;
  }
  if (added){ cues.sort((a,b) => a.start - b.start); render(); }
}

function render(){
  ui.empty.hidden = cues.length > 0;
  const mode = ui.showMode.value;
  ui.lines.innerHTML = '';
  for (let i = 0; i < cues.length; i++){
    const c = cues[i];
    const li = document.createElement('li');
    li.dataset.i = i;
    const en = mode !== 'src' && c.en ? `<div class="en">${esc(c.en)}</div>` : '';
    const src = (mode !== 'en' || !c.en) ? `<div class="src">${esc(c.src)}</div>` : '';
    li.innerHTML = `<span class="t">${fmt(c.start)}</span><div class="txt">${en}${src}</div>`;
    li.onclick = () => player?.seek(c.start);
    ui.lines.appendChild(li);
  }
  ui.lines.scrollTop = ui.lines.scrollHeight;
}

const esc = (s) => s.replace(/[&<>]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));

function cueAt(t){
  let lo = 0, hi = cues.length - 1, best = null;
  while (lo <= hi){
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t){ best = cues[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return best && t <= best.end + 0.6 ? best : null;
}

let shown = null;
function paint(){
  const t = player ? player.time() : 0;
  const c = cueAt(t) || (capturing ? cues[cues.length - 1] : null);
  if (c !== shown){
    shown = c;
    const mode = ui.showMode.value;
    ui.capPrimary.textContent   = !c ? '' : (mode === 'src' ? c.src : (c.en || c.src));
    ui.capSecondary.textContent = !c || mode !== 'both' ? '' : (c.en ? c.src : '');
    for (const li of ui.lines.children) li.classList.remove('active');
    if (c){
      const li = ui.lines.children[cues.indexOf(c)];
      if (li){ li.classList.add('active'); li.scrollIntoView({block:'nearest'}); }
    }
  }
  requestAnimationFrame(paint);
}
requestAnimationFrame(paint);

/* ---------------------------------------------------------------- status */
function say(text, cls = ''){ ui.status.textContent = text; ui.status.className = 'status ' + cls; }
function bar(pct){
  ui.progressWrap.hidden = pct == null;
  if (pct != null) ui.progressFill.style.width = pct + '%';
}

/* ---------------------------------------------------------------- audio */
let worker = null, ctx = null, stream = null, node = null;
let capturing = false, inFlight = false, ready = false;
let chunks = [], baseSample = 0, total = 0, nextStart = 0, seq = 0;
let sync = [];

const WORKLET = `
class Cap extends AudioWorkletProcessor {
  constructor(){ super(); this.b = new Float32Array(4096); this.n = 0; }
  process(inputs){
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++){
      this.b[this.n++] = ch[i];
      if (this.n === this.b.length){ this.port.postMessage(this.b.slice()); this.n = 0; }
    }
    return true;
  }
}
registerProcessor('cap', Cap);`;

function readRange(from, to){
  const out = new Float32Array(to - from);
  let pos = 0, idx = baseSample;
  for (const c of chunks){
    const s0 = idx, s1 = idx + c.length; idx = s1;
    if (s1 <= from) continue;
    if (s0 >= to) break;
    const a = Math.max(from, s0) - s0, b = Math.min(to, s1) - s0;
    out.set(c.subarray(a, b), pos); pos += b - a;
  }
  return out;
}

function trim(before){
  while (chunks.length && baseSample + chunks[0].length <= before){
    baseSample += chunks[0].length; chunks.shift();
  }
}

function sampleToVideo(abs){
  if (!sync.length) return 0;
  let lo = 0, hi = sync.length - 1, best = sync[0];
  while (lo <= hi){
    const mid = (lo + hi) >> 1;
    if (sync[mid].s <= abs){ best = sync[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return Math.max(0, best.v + (abs - best.s) / rate);
}

function pump(){
  while (true){
  if (!capturing || !ready || inFlight) return;

  // If inference has fallen far behind the audio, jump to the live edge.
  if (total - nextStart > WIN + 3 * STRIDE){
    nextStart = Math.max(nextStart, total - WIN);
    say('Skipping ahead to keep up — try the Base model.', 'err');
  }
  if (total < nextStart + WIN) return;

  const from = nextStart, to = from + WIN;
  const audio = to16k(readRange(from, to), rate);
  nextStart += STRIDE;
  trim(nextStart);

  let sum = 0;
  for (let i = 0; i < audio.length; i++) sum += audio[i] * audio[i];
  if (Math.sqrt(sum / audio.length) < SILENCE) continue;        // silence: don't wake the model

  inFlight = true;
  worker.postMessage({
    type:'audio', audio, id: ++seq, t0: sampleToVideo(from),
    lang: ui.srcLang.value, model: ui.model.value,
    translate: ui.showMode.value !== 'src',
  }, [audio.buffer]);
  return;
  }
}

async function start(){
  ui.startBtn.disabled = true;
  const src = ui.source.value;

  try{
    if (src === 'tab'){
      if (!navigator.mediaDevices?.getDisplayMedia)
        throw new Error('This browser cannot capture tab audio. Switch "Audio from" to Microphone.');
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: { echoCancellation:false, noiseSuppression:false, autoGainControl:false },
      });
      stream.getVideoTracks().forEach(t => t.stop());
      if (!stream.getAudioTracks().length){
        stream.getTracks().forEach(t => t.stop());
        throw new Error('Shared without audio — reshare and tick "Also share tab audio".');
      }
    } else {
      // Echo cancellation and noise suppression would strip out the very thing we want to
      // hear: sound coming from a speaker. Both off; leave gain control on for quiet rooms.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation:false, noiseSuppression:false, autoGainControl:true },
      });
    }
  }catch(e){
    ui.startBtn.disabled = false;
    const m = String(e?.message || e);
    if (/denied|NotAllowed/i.test(m))
      return say(src === 'mic' ? 'Microphone permission denied.' : 'Screen share cancelled.', 'err');
    return say(m, 'err');
  }

  stream.getAudioTracks()[0].onended = stop;

  ctx = new AudioContext({ sampleRate: SR });
  await ctx.resume();                                  // iOS starts contexts suspended
  rate   = ctx.sampleRate;                             // Safari may hand back 44100/48000
  WIN    = Math.round(WIN_S * rate);
  STRIDE = Math.round(STRIDE_S * rate);

  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type:'text/javascript'})));
  node = new AudioWorkletNode(ctx, 'cap');
  node.port.onmessage = (e) => { chunks.push(e.data); total += e.data.length; pump(); };

  const sink = ctx.createGain();
  sink.gain.value = 0;                                 // silent, but the graph must reach output
  ctx.createMediaStreamSource(stream).connect(node);
  node.connect(sink); sink.connect(ctx.destination);

  chunks = []; baseSample = 0; total = 0; nextStart = 0; sync = [];
  sync.push({ s:0, v: player ? player.time() : 0 });
  clearInterval(start._poll);
  start._poll = setInterval(() => {
    if (capturing && player) sync.push({ s: total, v: player.time() });
    if (sync.length > 600) sync.splice(0, 200);
  }, 250);

  capturing = true;
  ui.stopBtn.hidden = false;
  ui.startBtn.hidden = true;
  ui.startBtn.disabled = false;

  boot();
}

function boot(){
  if (worker) worker.terminate();
  ready = false; inFlight = false;
  worker = new Worker('./worker.js', { type:'module' });
  worker.onmessage = ({data:m}) => {
    if (m.type === 'progress'){ bar(m.pct); if (m.pct >= 100) say('Warming up…'); }
    else if (m.type === 'status') say(m.text);
    else if (m.type === 'ready'){ ready = true; bar(null); say('Listening — captions appear a few seconds behind.', 'live'); pump(); }
    else if (m.type === 'device'){ ui.engine.textContent = 'CPU only'; ui.engine.className = 'pill warn'; }
    else if (m.type === 'error'){ say('Model failed: ' + m.text, 'err'); bar(null); }
    else if (m.type === 'cues'){
      inFlight = false;
      if (m.cues?.length) addCues(m.cues);
      pump();
    }
  };
  say('Downloading model (once — then it is cached)…');
  worker.postMessage({
    type:'init', model: ui.model.value, gpu: hasGPU,
    lang: ui.srcLang.value, translate: ui.showMode.value !== 'src',
  });
}

function stop(){
  capturing = false;
  clearInterval(start._poll);
  stream?.getTracks().forEach(t => t.stop());
  try{ node?.disconnect(); ctx?.close(); }catch{}
  worker?.terminate(); worker = null; ready = false; inFlight = false;
  ui.stopBtn.hidden = true; ui.startBtn.hidden = false; ui.startBtn.disabled = false;
  bar(null);
  say(cues.length ? 'Stopped. Rewind the video — the transcript now tracks it exactly.' : 'Stopped.');
}

/* ---------------------------------------------------------------- export */
const pad = (n, w = 2) => String(n).padStart(w, '0');
function srtTime(t){
  const ms = Math.round(t * 1000);
  return `${pad(ms/3600000|0)}:${pad((ms/60000|0)%60)}:${pad((ms/1000|0)%60)},${pad(ms%1000,3)}`;
}
function download(name, body){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([body], {type:'text/plain;charset=utf-8'}));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

ui.srtBtn.onclick = () => {
  if (!cues.length) return;
  const mode = ui.showMode.value;
  const body = cues.map((c, i) => {
    const text = mode === 'src' ? c.src : (mode === 'en' ? (c.en || c.src) : [c.en, c.src].filter(Boolean).join('\n'));
    return `${i+1}\n${srtTime(c.start)} --> ${srtTime(Math.max(c.end, c.start + .8))}\n${text}\n`;
  }).join('\n');
  download('subtitles.srt', body);
};
ui.txtBtn.onclick = () => {
  if (!cues.length) return;
  download('transcript.txt', cues.map(c => `[${fmt(c.start)}] ${c.en || c.src}${c.en ? `\n         ${c.src}` : ''}`).join('\n'));
};
ui.clearBtn.onclick = () => { cues = []; lastEnd = -1; shown = null; render(); ui.capPrimary.textContent = ui.capSecondary.textContent = ''; };
ui.showMode.onchange = () => { shown = null; render(); };

/* ---------------------------------------------------------------- wiring */
ui.urlForm.onsubmit = async (e) => {
  e.preventDefault();
  const raw = ui.url.value.trim();
  if (!raw) return;
  ui.landing.hidden = true;
  ui.stage.hidden = false;
  try{ await mount(raw); say('Ready — press play, then Start captions.'); }
  catch(err){ say('Could not load that link: ' + err.message, 'err'); }
};
ui.startBtn.onclick = start;
ui.stopBtn.onclick = stop;
ui.source.value = canTab ? 'tab' : 'mic';
ui.source.onchange = syncSource;
syncSource();

// Deep link: ?v=<url> loads straight into the player.
const pre = new URLSearchParams(location.search).get('v');
if (pre){ ui.url.value = pre; ui.urlForm.requestSubmit(); }
