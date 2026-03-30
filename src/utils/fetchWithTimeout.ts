/**
 * fetch with an upper bound on wait time. Merges with an optional external AbortSignal.
 */
export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit & { timeoutMs?: number } = {}
): Promise<Response> {
  const { timeoutMs = 60_000, signal: outerSignal, ...rest } = init;
  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  if (outerSignal) {
    if (outerSignal.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    outerSignal.addEventListener("abort", onOuterAbort, { once: true });
  }

  timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(input, {
      ...rest,
      signal: controller.signal,
    });
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId);
    outerSignal?.removeEventListener("abort", onOuterAbort);
  }
}
