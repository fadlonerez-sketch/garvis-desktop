'use strict';

// Garvis service worker: keeps the app shell (page, script, styles, icons)
// available so the Home Screen app opens -- and says "PC offline" -- even when
// the PC can't be reached. API responses are never cached: task status and
// results always come live from the PC (the page keeps its own last-known
// task list for the offline view).

const SHELL = 'garvis-shell-v3';
const FILES = [
  '/',
  '/static/app.js',
  '/static/style.css',
  '/static/recorder-worklet.js',
  '/static/icon-180.png',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon-maskable-512.png',
  '/manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL).then((cache) => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== SHELL).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || url.pathname.startsWith('/api/')) {
    return; // straight to the network
  }
  if (!FILES.includes(url.pathname)) return;
  // Network first (so updates arrive), the cached shell when the PC is unreachable.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(SHELL).then((cache) => cache.put(event.request, copy));
        }
        return response;
      })
      .catch(() => caches.match(event.request)),
  );
});
