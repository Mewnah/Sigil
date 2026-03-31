/** Dev-only console logging; stripped from production user agents / minified dead-code elimination friendly. */
export function devLog(...args: unknown[]): void {
  if (import.meta.env.DEV) {
    console.log(...args);
  }
}
