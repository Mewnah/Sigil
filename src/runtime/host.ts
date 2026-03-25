/**
 * Typed accessors for host globals. New code can import these so tests or tooling can mock a single module.
 */
export function getConfig() {
  return window.Config;
}

export function getApiServer() {
  return window.ApiServer;
}

export function getApiClient() {
  return window.ApiClient;
}

export function getApiShared() {
  return window.ApiShared;
}
