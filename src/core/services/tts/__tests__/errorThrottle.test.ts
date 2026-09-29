import { describe, expect, it } from "vitest";
import { ErrorThrottle } from "../errorThrottle";

describe("ErrorThrottle", () => {
  function setup() {
    let now = 0;
    const throttle = new ErrorThrottle(15_000, 3_000, () => now);
    return { throttle, at: (t: number) => (now = t) };
  }

  it("shows the first error", () => {
    const { throttle } = setup();
    expect(throttle.shouldShow("server down")).toBe(true);
  });

  it("suppresses the same message for repeatMs", () => {
    const { throttle, at } = setup();
    expect(throttle.shouldShow("server down")).toBe(true);
    at(5_000);
    expect(throttle.shouldShow("server down")).toBe(false);
    at(14_999);
    expect(throttle.shouldShow("server down")).toBe(false);
    at(15_000);
    expect(throttle.shouldShow("server down")).toBe(true);
  });

  it("spaces out different messages by minGapMs", () => {
    const { throttle, at } = setup();
    expect(throttle.shouldShow("a")).toBe(true);
    at(1_000);
    expect(throttle.shouldShow("b")).toBe(false);
    at(3_000);
    expect(throttle.shouldShow("b")).toBe(true);
  });

  it("reset() lets the next error through", () => {
    const { throttle, at } = setup();
    expect(throttle.shouldShow("a")).toBe(true);
    at(10);
    throttle.reset();
    expect(throttle.shouldShow("a")).toBe(true);
  });

  it("forgets old messages when many distinct ones arrive", () => {
    const { throttle, at } = setup();
    for (let i = 0; i < 100; i++) {
      at(i * 20_000);
      expect(throttle.shouldShow(`error ${i}`)).toBe(true);
    }
  });
});
