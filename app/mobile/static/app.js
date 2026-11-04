'use strict';

// Garvis mobile client. Talks only to the Garvis server it was loaded from.
// All user/assistant/task text is rendered with textContent (never innerHTML).
// The PC is authoritative: this page only asks; Garvis on the PC validates,
// authorizes and executes.

const CLIENT_HEADERS = { 'X-Garvis-Client': '1' };
const MAX_RECORDING_MS = 60000;
const TARGET_SAMPLE_RATE = 16000;
const LONG_POLL_SECONDS = 20;
const TASK_CACHE_KEY = 'garvis.tasks.v1';
const UA = navigator.userAgent || '';
const IS_IOS = /iPhone|iPad|iPod/.test(UA) || (/Macintosh/.test(UA) && navigator.maxTouchPoints > 1);
const IS_ANDROID = /Android/.test(UA);

const state = {
  after: 0,
  polling: false,
  myTurns: new Map(), // turn_id -> { voice: bool }
  recording: null,
  starting: false,
  audioCtx: null,
  workletLoaded: false,
  playing: null,
  pendingIds: new Set(),
  sending: false,
  voiceInFlight: false,
  view: 'chat',
  online: false,
  lastSeen: 0,
  tasksVersion: -1,
  tasks: [],
  protocols: [],
  openTask: null, // task id shown in the sheet
  seenFinished: new Set(),
};

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      credentials: 'same-origin',
      cache: 'no-store',
      ...options,
      headers: { ...CLIENT_HEADERS, ...(options.headers || {}) },
    });
  } catch (err) {
    throw new ApiError('Cannot reach Garvis. Is the PC on, Garvis running, and the phone on the same network?', 0);
  }
  const type = response.headers.get('Content-Type') || '';
  const data = type.includes('application/json') ? await response.json().catch(() => null) : null;
  if (!response.ok) {
    throw new ApiError((data && data.error) || `Request failed (HTTP ${response.status}).`, response.status);
  }
  return data;
}

function requestId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, '');
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// One submission = one request id. A retry after a network failure reuses
// it, so the PC returns the task it already accepted instead of running the
// instruction twice.
async function postOnce(path, body) {
  const payload = JSON.stringify({ ...body, request_id: requestId() });
  const options = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload };
  try {
    return await api(path, options);
  } catch (err) {
    if (err.status !== 0) throw err;
    await sleep(1500);
    return api(path, options);
  }
}

// ------------------------------------------------------------------ UI bits
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function showBanner(text) {
  const banner = $('banner');
  banner.textContent = text;
  banner.hidden = !text;
  // Views move down under the banner instead of being covered by it.
  document.documentElement.style.setProperty('--banner-h', text ? `${banner.offsetHeight + 8}px` : '0px');
}

function setStatus(text) {
  $('status-text').textContent = text;
}

function setConnection(mode, busy) {
  const pill = $('status-pill');
  pill.className = `pill pill-${mode}`;
  $('pill-text').textContent = { offline: 'PC offline', unpaired: 'Not paired' }[mode] || (busy ? 'Working' : 'PC online');
  document.body.classList.toggle('busy', Boolean(busy) && mode !== 'offline');
  state.online = mode !== 'offline' && mode !== 'unpaired';
  if (state.online) state.lastSeen = Date.now();
}

function clock(seconds) {
  if (!seconds) return '';
  return new Date(seconds * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function ago(seconds) {
  if (!seconds) return '';
  const delta = Math.max(0, Date.now() / 1000 - seconds);
  if (delta < 60) return 'just now';
  if (delta < 3600) return `${Math.floor(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} h ago`;
  return new Date(seconds * 1000).toLocaleDateString();
}

function scrollToBottom() {
  const chat = $('chat-view');
  chat.scrollTop = chat.scrollHeight;
}

function addItem(className, text, meta) {
  const item = el('li', `msg ${className}`);
  if (meta) item.appendChild(el('span', 'msg-meta', meta));
  item.appendChild(document.createTextNode(text));
  $('messages').appendChild(item);
  return item;
}

function showError(text) {
  addItem('msg-error', text);
  scrollToBottom();
}

function setView(view) {
  const paired = view !== 'pair';
  $('pair-view').hidden = paired;
  $('tabbar').hidden = !paired;
  if (!paired) {
    for (const id of ['chat-view', 'tasks-view', 'protocols-view', 'composer', 'confirmations']) $(id).hidden = true;
    return;
  }
  state.view = view;
  $('chat-view').hidden = view !== 'chat';
  $('composer').hidden = view !== 'chat';
  $('tasks-view').hidden = view !== 'tasks';
  $('protocols-view').hidden = view !== 'protocols';
  for (const tab of document.querySelectorAll('.tab[data-view]')) {
    tab.classList.toggle('active', tab.dataset.view === view);
  }
  if (view === 'tasks') refreshTasks(true);
  if (view === 'protocols') refreshProtocols();
  if (view === 'chat') scrollToBottom();
}

// ------------------------------------------------------------ event render
const OUTCOME_TEXT = {
  approved: 'Approved',
  denied: 'Denied',
  timeout: 'Timed out (not run)',
  cancelled: 'Cancelled',
};

const STATUS_TEXT = {
  queued: 'Queued',
  running: 'Running',
  waiting_confirmation: 'Needs approval',
  cancelling: 'Stopping',
  completed: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
};

function renderEvent(event, live) {
  switch (event.kind) {
    case 'user':
      addItem('msg-user', event.text, event.source === 'mobile' ? null : 'On the PC');
      break;
    case 'assistant': {
      const item = addItem('msg-assistant', event.text);
      if (event.turn_id) { // a reply to a turn sent from a phone: playable here
        const play = el('button', 'play', '🔊 Play');
        play.type = 'button';
        play.addEventListener('click', () => {
          unlockAudio();
          playReply(event.turn_id);
        });
        item.appendChild(document.createElement('br'));
        item.appendChild(play);
      }
      break;
    }
    case 'tool_started':
      addItem('msg-activity', `⚙ ${event.name}…`);
      break;
    case 'tool_finished':
      addItem('msg-activity', event.success ? `✓ ${event.name}` : `✗ ${event.name}${event.error ? ` — ${event.error}` : ''}`);
      break;
    case 'confirmation_requested':
      addItem('msg-activity', `Approval needed: ${event.tool_name}`);
      break;
    case 'confirmation_resolved':
      addItem('msg-activity', `${OUTCOME_TEXT[event.outcome] || event.outcome}`);
      break;
    case 'error':
      addItem('msg-error', event.message);
      break;
    case 'turn_finished': {
      if (event.status === 'cancelled') addItem('msg-activity', 'Task cancelled.');
      // Auto-play only replies to this page's own voice turns. The upload's
      // response may still be in flight when a fast reply arrives.
      const turn = state.myTurns.get(event.turn_id);
      const mine = turn ? turn.voice : state.voiceInFlight;
      if (live && mine && event.turn_kind === 'voice' && event.has_reply) {
        playReply(event.turn_id);
      }
      break;
    }
    default:
      break;
  }
}

function renderConfirmations(pending) {
  const container = $('confirmations');
  const ids = new Set(pending.map((p) => p.id));
  for (const card of [...container.children]) {
    if (!ids.has(card.dataset.id)) card.remove();
  }
  for (const request of pending) {
    if (state.pendingIds.has(request.id) && container.querySelector(`[data-id="${CSS.escape(request.id)}"]`)) continue;
    container.appendChild(buildConfirmationCard(request));
  }
  state.pendingIds = ids;
  container.hidden = pending.length === 0;
}

function buildConfirmationCard(request) {
  const card = el('article', 'card');
  card.dataset.id = request.id;
  card.append(el('h3', null, `Allow “${request.tool_name}”?`), el('p', null, request.reason));
  const entries = Object.entries(request.arguments || {});
  if (entries.length) {
    const list = el('dl');
    for (const [key, value] of entries) list.append(el('dt', null, key), el('dd', null, value));
    card.appendChild(list);
  }
  const actions = el('div', 'actions');
  const deny = el('button', 'deny', 'Deny');
  const approve = el('button', 'approve', 'Approve');
  deny.type = 'button';
  approve.type = 'button';
  const answer = async (approved) => {
    deny.disabled = true;
    approve.disabled = true;
    try {
      await api(`/api/confirmations/${encodeURIComponent(request.id)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      });
    } catch (err) {
      showError(err.status === 404 ? 'That request was already answered or timed out.' : err.message);
    }
    card.remove();
    if (!$('confirmations').children.length) $('confirmations').hidden = true;
  };
  deny.addEventListener('click', () => answer(false));
  approve.addEventListener('click', () => answer(true));
  actions.append(deny, approve);
  card.appendChild(actions);
  return card;
}

// ------------------------------------------------------------------ tasks
function cacheTasks(tasks) {
  try {
    localStorage.setItem(TASK_CACHE_KEY, JSON.stringify({ at: Date.now(), tasks }));
  } catch (err) { /* private mode or full: the cache is only a convenience */ }
}

function cachedTasks() {
  try {
    return JSON.parse(localStorage.getItem(TASK_CACHE_KEY) || 'null');
  } catch (err) {
    return null;
  }
}

function taskMeta(task) {
  const origin = task.origin === 'mobile' ? (task.device || 'phone') : 'PC';
  const kind = task.kind === 'protocol' ? 'Protocol' : task.kind === 'chat' ? 'Chat' : 'Task';
  const when = task.finished_at ? `finished ${ago(task.finished_at)}` : task.started_at ? `started ${clock(task.started_at)}`
    : `queued ${clock(task.created_at)}`;
  return `${kind} · from ${origin} · ${when}`;
}

function renderTasks(tasks, stale) {
  const list = $('task-list');
  list.replaceChildren();
  for (const task of tasks) {
    const card = el('li', `card status-${task.status}`);
    card.tabIndex = 0;
    const row = el('div', 'card-row');
    row.append(el('p', 'card-title', task.title), el('span', `chip chip-${task.status}`, STATUS_TEXT[task.status] || task.status));
    card.append(row, el('p', 'card-meta', taskMeta(task)));
    const line = task.note || (task.status === 'failed' || task.status === 'interrupted' ? task.error : task.result_preview);
    if (line) card.appendChild(el('p', 'card-text', line));
    if (task.progress && task.progress.total && !['completed', 'cancelled'].includes(task.status)) {
      const bar = el('div', 'progress');
      const fill = el('span');
      fill.style.width = `${Math.round((100 * Math.max(0, task.progress.step - (task.status === 'running' ? 0.5 : 0))) / task.progress.total)}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
    }
    card.addEventListener('click', () => openTask(task.id));
    list.appendChild(card);
  }
  $('tasks-empty').hidden = tasks.length > 0;
  $('tasks-stamp').textContent = stale ? `Offline — as of ${clock(stale / 1000)}` : '';
  const active = tasks.filter((t) => ['queued', 'running', 'waiting_confirmation', 'cancelling'].includes(t.status)).length;
  $('tasks-badge').hidden = active === 0;
  $('tasks-badge').textContent = String(active);
}

async function refreshTasks(force) {
  try {
    const data = await api('/api/tasks');
    state.tasks = data.tasks;
    state.tasksVersion = data.version;
    cacheTasks(data.tasks);
    renderTasks(data.tasks, 0);
    if (state.openTask) refreshOpenTask();
  } catch (err) {
    if (err.status === 401) return;
    if (force) {
      const cached = cachedTasks();
      if (cached) renderTasks(cached.tasks, cached.at);
    }
  }
}

function stepMark(status) {
  return { ok: '✓', failed: '✕', timeout: '!', cancelled: '–', skipped: '–' }[status] || '';
}

function buildTaskDetail(task) {
  const body = $('sheet-body');
  body.replaceChildren();
  const status = el('p', 'card-meta', `${STATUS_TEXT[task.status] || task.status} · ${taskMeta(task)}`);
  body.appendChild(status);
  if (task.note) body.appendChild(el('p', 'card-text', task.note));
  if (task.text && task.kind !== 'protocol') {
    body.append(el('h4', null, 'Instruction'), el('p', null, task.text));
  }
  if (task.steps && task.steps.length) {
    body.appendChild(el('h4', null, task.kind === 'protocol' ? 'Steps' : 'Actions'));
    const steps = el('ul', 'steps');
    for (const step of task.steps) {
      const item = el('li', step.status);
      const label = el('div', null, step.label);
      if (step.message) label.appendChild(el('small', null, step.message));
      item.append(el('span', 'mark', stepMark(step.status)), label);
      steps.appendChild(item);
    }
    body.appendChild(steps);
  }
  if (task.opened && task.opened.length) {
    body.appendChild(el('h4', null, 'Shown on the PC'));
    const list = el('ul', 'links');
    for (const item of task.opened) {
      const li = el('li', null, item.title);
      if (item.detail) li.appendChild(el('small', null, item.detail));
      list.appendChild(li);
    }
    body.appendChild(list);
  }
  if (task.result) body.append(el('h4', null, 'Result'), el('p', null, task.result));
  if (task.error && task.error !== task.result) {
    const error = el('p', null, task.error);
    error.style.color = 'var(--error)';
    body.append(el('h4', null, 'Problem'), error);
  }
  if (task.sources && task.sources.length) {
    body.appendChild(el('h4', null, 'Sources'));
    const list = el('ul', 'links');
    for (const source of task.sources) {
      const li = el('li');
      const link = el('a', null, source.title || source.url);
      link.href = source.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      li.append(link, el('small', null, source.publisher || source.url));
      list.appendChild(li);
    }
    body.appendChild(list);
  }
  const actions = $('sheet-actions');
  actions.replaceChildren();
  if (task.cancellable) {
    const cancel = el('button', 'danger', task.status === 'queued' ? 'Cancel' : 'Stop this task');
    cancel.type = 'button';
    cancel.addEventListener('click', async () => {
      cancel.disabled = true;
      try {
        await api(`/api/tasks/${encodeURIComponent(task.id)}/cancel`, { method: 'POST' });
        await refreshOpenTask();
      } catch (err) {
        cancel.disabled = false;
        showBanner(err.message);
        setTimeout(() => showBanner(''), 4000);
      }
    });
    actions.appendChild(cancel);
  }
}

async function refreshOpenTask() {
  if (!state.openTask) return;
  try {
    const task = await api(`/api/tasks/${encodeURIComponent(state.openTask)}`);
    if (state.openTask === task.id) {
      $('sheet-title').textContent = task.title;
      buildTaskDetail(task);
    }
  } catch (err) {
    if (err.status === 404) closeSheet();
  }
}

function openSheet(title) {
  if ($('sheet').hidden) history.pushState({ garvisSheet: true }, '');
  document.body.classList.add('sheet-open');
  $('sheet-title').textContent = title;
  $('sheet-body').replaceChildren();
  $('sheet-actions').replaceChildren();
  $('sheet').hidden = false;
  $('sheet-backdrop').hidden = false;
}

function closeSheet(fromHistory = false) {
  state.openTask = null;
  const wasOpen = !$('sheet').hidden;
  document.body.classList.remove('sheet-open');
  $('sheet').hidden = true;
  $('sheet-backdrop').hidden = true;
  if (wasOpen && !fromHistory && history.state && history.state.garvisSheet) {
    state.ignorePop = true; // our own step back: its popstate must not close a sheet opened right after
    history.back();
  }
}

function openTask(taskId) {
  const summary = state.tasks.find((t) => t.id === taskId);
  openSheet(summary ? summary.title : 'Task');
  state.openTask = taskId;
  if (!state.online) {
    $('sheet-body').appendChild(el('p', 'card-text', 'The PC is offline — details appear when it is reachable again.'));
    return;
  }
  refreshOpenTask();
}

// -------------------------------------------------------------- protocols
function renderProtocols(protocols) {
  const list = $('protocol-list');
  list.replaceChildren();
  for (const protocol of protocols) {
    const card = el('li', 'card');
    card.appendChild(el('p', 'card-title', protocol.name));
    if (protocol.description) card.appendChild(el('p', 'card-text', protocol.description));
    const steps = el('ul', 'step-preview');
    for (const step of protocol.steps.slice(0, 6)) steps.appendChild(el('li', null, step));
    if (protocol.steps.length > 6) steps.appendChild(el('li', null, `+ ${protocol.steps.length - 6} more`));
    card.appendChild(steps);
    if (protocol.problem) card.appendChild(el('p', 'problem', protocol.problem));
    const run = el('button', 'primary run', 'Run on the PC');
    run.type = 'button';
    run.disabled = !protocol.runnable || !state.online;
    run.addEventListener('click', () => confirmRun(protocol));
    card.appendChild(run);
    list.appendChild(card);
  }
  $('protocols-empty').hidden = protocols.length > 0;
}

async function refreshProtocols() {
  try {
    const data = await api('/api/protocols');
    state.protocols = data.protocols;
    renderProtocols(data.protocols);
  } catch (err) {
    if (err.status !== 401) renderProtocols(state.protocols);
  }
}

function confirmRun(protocol) {
  openSheet(`Run “${protocol.name}”?`);
  const body = $('sheet-body');
  body.appendChild(el('p', 'card-text', 'Garvis will do this on the PC now. Steps that need approval will still ask.'));
  const steps = el('ul', 'steps');
  protocol.steps.forEach((step, i) => {
    const item = el('li');
    item.append(el('span', 'mark', String(i + 1)), el('div', null, step));
    steps.appendChild(item);
  });
  body.appendChild(steps);
  const actions = $('sheet-actions');
  const cancel = el('button', 'ghost', 'Not now');
  const go = el('button', 'primary', 'Run');
  cancel.type = 'button';
  go.type = 'button';
  cancel.addEventListener('click', () => closeSheet());
  go.addEventListener('click', async () => {
    go.disabled = true;
    try {
      const task = await postOnce(`/api/protocols/${encodeURIComponent(protocol.id)}/run`, {});
      setView('tasks');
      openTask(task.id); // reuses the open sheet (and its history entry) for the task's progress
    } catch (err) {
      go.disabled = false;
      body.appendChild(el('p', 'problem', err.message));
    }
  });
  actions.append(cancel, go);
}

// ------------------------------------------------------------- long-poll
async function pollLoop() {
  if (state.polling) return;
  state.polling = true;
  let first = true;
  while (state.polling) {
    try {
      const wait = first ? 0 : LONG_POLL_SECONDS;
      const data = await api(`/api/events?after=${state.after}&wait=${wait}`);
      if (data.reset) {
        $('messages').replaceChildren();
        state.after = 0;
        continue;
      }
      const nearBottom = isNearBottom();
      for (const event of data.events) renderEvent(event, !first);
      state.after = data.last_seq;
      if (data.events.length && (nearBottom || first)) scrollToBottom();
      renderConfirmations(data.pending_confirmations || []);
      setConnection(data.busy ? 'busy' : 'online', data.busy);
      setStatus(data.busy ? data.status || 'Working…' : 'Ready');
      showBanner('');
      if (data.tasks_version !== state.tasksVersion) refreshTasks(false);
      first = false;
    } catch (err) {
      if (err.status === 401) {
        state.polling = false;
        showPairing();
        return;
      }
      setConnection('offline', false);
      const seen = state.lastSeen ? ` Last reached ${clock(state.lastSeen / 1000)}.` : '';
      setStatus('Reconnecting…');
      showBanner(`Garvis on the PC can't be reached right now.${seen} Tasks it already accepted keep running there; `
        + "new messages and protocol runs can't be sent until it is back.");
      const cached = cachedTasks();
      if (cached && state.view === 'tasks') renderTasks(cached.tasks, cached.at);
      await sleep(3000);
    }
  }
}

function isNearBottom() {
  const chat = $('chat-view');
  return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
}

// ---------------------------------------------------------------- pairing
function showPairing() {
  setView('pair');
  setConnection('unpaired', false); // the PC answered: it's reachable, this phone just isn't paired
  setStatus('Not paired');
  $('pair-code').focus();
}

async function onPairSubmit(event) {
  event.preventDefault();
  $('pair-error').textContent = '';
  try {
    await api('/api/pair', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: $('pair-code').value }),
    });
    $('pair-code').value = '';
    showChat();
  } catch (err) {
    $('pair-error').textContent = err.message;
  }
}

async function onUnpair() {
  if (!window.confirm('Unpair this phone? You will need a new code from the PC.')) return;
  try {
    await api('/api/unpair', { method: 'POST' });
  } catch (err) {
    // Already unpaired or offline: either way, show the pairing screen.
  }
  state.polling = false;
  $('messages').replaceChildren();
  state.after = 0;
  try { localStorage.removeItem(TASK_CACHE_KEY); } catch (err) { /* ignore */ }
  showPairing();
}

function showChat() {
  setView('chat');
  setAudioSessionType('playback');
  pollLoop();
}

// ------------------------------------------------------------ text input
async function onSend(event) {
  event.preventDefault();
  const input = $('message-input');
  const text = input.value.trim();
  if (!text || state.sending) return;
  state.sending = true;
  $('send-button').disabled = true;
  try {
    const result = await postOnce('/api/messages', { text });
    state.myTurns.set(result.turn_id, { voice: false });
    input.value = '';
    autoGrow();
  } catch (err) {
    showError(err.status === 0 ? `Not sent — ${err.message}` : err.message);
  } finally {
    state.sending = false;
    $('send-button').disabled = false;
  }
}

function autoGrow() {
  const input = $('message-input');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

// ------------------------------------------------------------------ audio
function setAudioSessionType(type) {
  // Newer Safari: 'playback' plays through the speaker even with the ring/silent
  // switch on; 'play-and-record' is needed while the microphone is open.
  try {
    if (navigator.audioSession) navigator.audioSession.type = type;
  } catch (err) {
    // Older iOS: not supported, nothing to do.
  }
}

function unlockAudio() {
  // Must run synchronously inside a tap: iOS only lets a page start audio
  // from a user gesture, after which the same AudioContext stays usable.
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  if (!state.audioCtx || state.audioCtx.state === 'closed') {
    state.audioCtx = new Ctx();
    state.workletLoaded = false;
  }
  if (state.audioCtx.state !== 'running') {
    state.audioCtx.resume().catch(() => {});
  }
  return state.audioCtx;
}

function stopPlayback() {
  if (state.playing) {
    try {
      state.playing.stop();
    } catch (err) {
      // already stopped
    }
    state.playing = null;
  }
  $('stop-audio').hidden = true;
  refreshVoiceState();
}

async function playReply(turnId) {
  try {
    const response = await fetch(`/api/turns/${encodeURIComponent(turnId)}/speech`, {
      credentials: 'same-origin',
      cache: 'no-store',
      headers: CLIENT_HEADERS,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => null);
      throw new Error((data && data.error) || `HTTP ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    const ctx = state.audioCtx;
    if (!ctx || ctx.state !== 'running') {
      throw new Error('The browser blocked automatic playback. Tap 🔊 Play on the reply.');
    }
    const buffer = await new Promise((resolve, reject) => ctx.decodeAudioData(bytes, resolve, reject));
    stopPlayback();
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      if (state.playing === source) {
        state.playing = null;
        $('stop-audio').hidden = true;
        refreshVoiceState();
      }
    };
    source.start();
    state.playing = source;
    $('stop-audio').hidden = false;
    refreshVoiceState('Speaking…');
  } catch (err) {
    showError(`Could not play the reply: ${err.message || err}`);
  }
}

function refreshVoiceState(label) {
  const text = label || (state.recording ? 'Recording… tap the mic again to send' : '');
  $('voice-label').textContent = text;
  $('voice-state').hidden = !text && $('stop-audio').hidden;
}

function micErrorMessage(err) {
  const name = err && err.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    if (IS_ANDROID) {
      return 'Microphone access was denied. Tap the icon left of the address (or ⋮ › Site settings) › '
        + 'Permissions › Microphone › Allow. Also check Android Settings › Apps › Chrome › Permissions › Microphone.';
    }
    return 'Microphone access was denied. Allow it in Settings › Apps › Safari › Microphone '
      + '(or tap “aA” › Website Settings › Microphone), then try again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No microphone is available.';
  }
  if (name === 'NotReadableError' || name === 'AbortError') {
    return 'The microphone is busy (a call or another app may be using it).';
  }
  return `Could not start recording: ${(err && err.message) || err}`;
}

async function startRecording() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(window.isSecureContext
      ? 'This browser cannot record audio.'
      : 'The microphone needs a trusted HTTPS connection. Install and trust the Garvis certificate (see the README), then reload.');
    return;
  }
  const ctx = unlockAudio();
  if (!ctx) {
    showError('This browser has no Web Audio support, so it cannot record.');
    return;
  }
  stopPlayback();
  state.starting = true;
  setAudioSessionType('play-and-record');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    state.starting = false;
    setAudioSessionType('playback');
    showError(micErrorMessage(err));
    return;
  }

  try {
    await ctx.resume();
    const source = ctx.createMediaStreamSource(stream);
    const chunks = [];
    let node;
    if (ctx.audioWorklet && window.AudioWorkletNode) {
      if (!state.workletLoaded) {
        await ctx.audioWorklet.addModule('/static/recorder-worklet.js');
        state.workletLoaded = true;
      }
      node = new AudioWorkletNode(ctx, 'garvis-recorder');
      node.port.onmessage = (event) => chunks.push(event.data);
    } else {
      node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (event) => chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    }
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);
    state.recording = {
      stream, source, node, mute, chunks,
      sampleRate: ctx.sampleRate,
      timer: setTimeout(() => stopRecording(true), MAX_RECORDING_MS),
    };
    const mic = $('mic-button');
    mic.classList.add('recording');
    mic.setAttribute('aria-label', 'Stop recording and send');
    refreshVoiceState();
  } catch (err) {
    stream.getTracks().forEach((track) => track.stop());
    setAudioSessionType('playback');
    showError(`Could not start recording: ${err.message || err}`);
  } finally {
    state.starting = false;
  }
}

async function stopRecording(timedOut = false) {
  const rec = state.recording;
  if (!rec) return;
  state.recording = null;
  clearTimeout(rec.timer);
  if (rec.node.port) {
    rec.node.port.postMessage('flush');
    await sleep(150); // let the last batch arrive
    rec.node.port.onmessage = null;
  } else {
    rec.node.onaudioprocess = null;
  }
  rec.source.disconnect();
  rec.node.disconnect();
  rec.mute.disconnect();
  rec.stream.getTracks().forEach((track) => track.stop());
  setAudioSessionType('playback');
  const mic = $('mic-button');
  mic.classList.remove('recording');
  mic.setAttribute('aria-label', 'Start recording');

  const samples = concatSamples(rec.chunks);
  if (samples.length < rec.sampleRate * 0.4) {
    refreshVoiceState();
    showError('That recording was too short. Tap the mic, speak, then tap again.');
    return;
  }
  if (timedOut) {
    showError('Recording stopped at the 60-second limit; sending what was recorded.');
  }
  const targetRate = Math.min(TARGET_SAMPLE_RATE, rec.sampleRate);
  const wav = encodeWav(downsample(samples, rec.sampleRate, targetRate), targetRate);

  mic.disabled = true;
  refreshVoiceState('Transcribing on the PC…');
  state.voiceInFlight = true;
  try {
    const result = await api('/api/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav,
    });
    state.myTurns.set(result.turn_id, { voice: true });
  } catch (err) {
    showError(err.message);
  } finally {
    state.voiceInFlight = false;
    mic.disabled = false;
    refreshVoiceState();
  }
}

function concatSamples(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function downsample(input, inRate, outRate) {
  if (outRate >= inRate) return input;
  const ratio = inRate / outRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function onMicClick() {
  if (state.recording) {
    stopRecording();
  } else if (!state.starting) {
    startRecording();
  }
}

// ------------------------------------------------------------------- boot
async function boot() {
  $('pair-form').addEventListener('submit', onPairSubmit);
  $('message-form').addEventListener('submit', onSend);
  $('message-input').addEventListener('input', autoGrow);
  $('message-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) onSend(event);
  });
  $('mic-button').addEventListener('click', onMicClick);
  $('stop-audio').addEventListener('click', stopPlayback);
  $('menu-button').addEventListener('click', onUnpair);
  $('sheet-close').addEventListener('click', () => closeSheet());
  $('sheet-backdrop').addEventListener('click', () => closeSheet());
  window.addEventListener('popstate', () => {
    if (state.ignorePop) {
      state.ignorePop = false;
      // Our own step back landed after a new sheet opened: give that sheet its entry back.
      if (!$('sheet').hidden) history.pushState({ garvisSheet: true }, '');
      return;
    }
    if (!$('sheet').hidden) closeSheet(true); // Android back gesture: close the sheet, stay in the app
  });
  for (const tab of document.querySelectorAll('.tab[data-view]')) {
    tab.addEventListener('click', () => setView(tab.dataset.view));
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.view === 'tasks') refreshTasks(true);
  });

  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {});
  }
  if (!window.isSecureContext) {
    showBanner('This page is not on a trusted HTTPS connection, so the microphone and app install are unavailable. '
      + (IS_ANDROID ? "Install the Garvis certificate in Android Settings (see the PC's Phone dialog)."
        : 'Install and trust the Garvis certificate (see the README).'));
  }

  for (;;) {
    try {
      const session = await api('/api/session');
      if (session.paired) showChat();
      else showPairing();
      return;
    } catch (err) {
      setConnection('offline', false);
      setStatus('Cannot reach Garvis — retrying…');
      const cached = cachedTasks();
      if (cached) {
        $('tabbar').hidden = false;
        state.view = 'tasks';
        $('tasks-view').hidden = false;
        renderTasks(cached.tasks, cached.at);
        showBanner("Garvis on the PC can't be reached. Showing the last known tasks.");
      }
      await sleep(3000);
    }
  }
}

document.addEventListener('DOMContentLoaded', boot);
