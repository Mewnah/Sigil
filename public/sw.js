/*
 * Kill-switch service worker. Nothing in Sigil registers this file.
 *
 * Why it exists: Curses and early Sigil builds registered a caching service worker at /sw.js in the
 * browser that opened /client (an OBS browser source keeps its profile between runs). That worker
 * keeps serving the old app from its cache, so such a source kept showing the old "connecting..."
 * page, and Sigil's own page never got to run. The browser re-checks /sw.js by itself when the source
 * loads; this file is what that check now finds. It has no fetch handler, so it never answers a
 * request. In order:
 *
 *   install   skipWaiting(), and every cache is deleted at once. The old worker is still the active
 *             one until this one takes over, but with its cache empty it falls back to the network,
 *             which is Sigil's current page. Without this the old page's own `updatefound` handler
 *             (`await registration.update(); location.reload()`) reloads straight back into the
 *             cached old page, and with several windows of this server open at once (/client plus a
 *             captions page or a dock) the hand-off then stalled until the next navigation
 *             (scripts/check-sw-kill-switch.mjs, three windows, measured).
 *   activate  unregisters this worker, then reloads the windows this registration controlled (the
 *             pages the old worker was showing) from the network. Windows it never controlled, such
 *             as a captions page opened before the old worker existed, are left alone: no
 *             clients.claim(), no includeUncontrolled. The reloads are issued, not awaited: a
 *             navigation that stalls must not hold the activate event open. A last sweep deletes
 *             any cache that appeared in between.
 *
 * Every storage call is bounded by a timeout so a slow one cannot hold install or activate open;
 * a stuck hand-off is then finished by the source's next load.
 */
var STEP_TIMEOUT_MS = 2000;

/** Resolves with `promise`, or with undefined after `ms` if it has not settled by then. */
function within(promise, ms) {
  return Promise.race([
    promise,
    new Promise(function (resolve) {
      setTimeout(resolve, ms);
    }),
  ]);
}

function deleteAllCaches() {
  return caches
    .keys()
    .then(function (keys) {
      return Promise.all(
        keys.map(function (key) {
          return caches.delete(key);
        }),
      );
    })
    .catch(function () {});
}

self.addEventListener("install", function (event) {
  event.waitUntil(Promise.all([self.skipWaiting(), within(deleteAllCaches(), STEP_TIMEOUT_MS)]));
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    within(self.clients.matchAll({ type: "window" }), STEP_TIMEOUT_MS)
      .catch(function () {
        return [];
      })
      .then(function (windows) {
        return within(self.registration.unregister(), STEP_TIMEOUT_MS)
          .catch(function () {})
          .then(function () {
            (windows || []).forEach(function (client) {
              try {
                client.navigate(client.url).catch(function () {});
              } catch (error) {
                /* a window that cannot be navigated is fixed by its own next load */
              }
            });
            return within(deleteAllCaches(), STEP_TIMEOUT_MS);
          });
      }),
  );
});
