import fastEqual from "fast-deep-equal";
import { FC, memo, useCallback, useEffect, useRef } from "react";
import { subscribe, useSnapshot } from "valtio";
import { useGetState } from "../..";
import { TextEvent, TextEventSource, TextEventType } from "../../../types";
import {
  CaptionDom,
  CaptionKind,
  CaptionLine,
  CaptionUpdate,
  detachCaptionLines,
  findCaptionLine,
  isStaleUpdate,
  readCaptionMeta,
  RetiredCaptionLines,
  supersededCaptionLines,
} from "./revising";
import { Element_TextState } from "./schema";
import { buildStateStyle, elementStyle } from "./style";

type SentenceState = {
  line: CaptionLine<Node>
  element: HTMLElement
  /** Nothing of the line is left to animate (or it is not its turn yet: see `nextAnimationIndex`). */
  isAnimated: boolean
}

/*
 * Text reaches this element from STT, translation, Twitch chat and LAN peers, so it is never parsed
 * as HTML: text goes into text nodes, masks are spans with textContent and emotes are <img> elements
 * whose src is set as a property (the host webview has full Tauri IPC).
 *
 * Revising text (./revising.ts): an update pairs with its line by segmentId / id / revision and only
 * the part after what is still correct on screen is re-rendered, so a line never restarts when just
 * its tail changes. An interim line keeps the interim look; TextEvent.stableLength additionally marks
 * its uncommitted tail (`.tentative`).
 */
function setTextData(node: Text, data: string) {
  const old = node.data;
  if (data === old) return;
  if (data.startsWith(old)) node.appendData(data.slice(old.length));
  else if (old.startsWith(data)) node.deleteData(data.length, old.length - data.length);
  else node.data = data;
}

const DOM_CAPTIONS: CaptionDom<Node> = {
  text: data => document.createTextNode(data),
  span: (className, data) => {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = data;
    return span;
  },
  image: (className, url) => {
    const img = document.createElement("img");
    if (className) img.className = className;
    img.src = url;
    return img;
  },
  setData: (node, data) => {
    if (node.nodeType === Node.TEXT_NODE) {
      setTextData(node as Text, data);
      return;
    }
    const only = node.firstChild;
    if (only && only.nodeType === Node.TEXT_NODE && !only.nextSibling) setTextData(only as Text, data);
    else node.textContent = data;
  },
  setClass: (node, className) => {
    const el = node as Element;
    if ((el.getAttribute("class") ?? "") === className) return;
    if (className) el.className = className;
    else el.removeAttribute("class");
  },
  append: (parent, node) => { parent.appendChild(node); },
  remove: node => { (node as ChildNode).remove(); },
  replace: (oldNode, ...nodes) => { (oldNode as ChildNode).replaceWith(...nodes); },
};

class TextController {
  constructor(private id: string) { }
  private disposed: boolean = false;

  private boxElement!: HTMLElement
  private containerElement?: HTMLElement
  private scrollContainerElement!: HTMLElement
  private textElement!: HTMLElement
  private textEndElement!: HTMLSpanElement
  private baseStyleElement?: HTMLStyleElement;
  private stateStyleElement?: HTMLStyleElement;

  private textObserver?: ResizeObserver;

  private eventTextRef: any;
  private eventTextInputRef: any;
  private storeEventCancelToken?: () => void;
  private sceneChangeEventCancelToken?: () => void;
  private fileUrlsUnsub?: () => void;
  private peerMirrorUnsub?: () => void;

  private currentState?: Element_TextState;
  private sentenceQueue: SentenceState[] = [];
  /** Lines that left the screen, so a late update of one does not bring it back. */
  private retiredLines = new RetiredCaptionLines();

  private isPlayingAnimation = false;
  private animatingSentence: SentenceState | null = null;
  private nextAnimationIndex = 0;

  private activityTimerHandle: number = -1;
  private activityTimerDelayClearHandle: number = -1;
  private animationTimerHandle: number = -1;

  private get state(): Element_TextState {
    return this.currentState!;
  }

  private get profanityMask(): string {
    const mask = this.state.textProfanityMask;
    return typeof mask === "string" ? mask : "";
  }

  onShow() {
    if (this.state.soundEnable && this.state.soundFileOnShow) {
      window.ApiClient.sound.playFile(this.state.soundFileOnShow, { volume: this.state.soundVolume ?? 1 });
    }
  }
  onHide() {
    if (this.state.soundEnable && this.state.soundFileOnHide) {
      window.ApiClient.sound.playFile(this.state.soundFileOnHide, { volume: this.state.soundVolume ?? 1 });
    }
  }
  onNewSentence() { }

  onActivity() {
    try {
      const rect = this.textEndElement.getBoundingClientRect();

      if (this.state.animateEvent) {
        window.ApiShared.pubsub.publishLocally({ topic: `element.${this.id}` });
      }

      if (rect && this.state.particlesEnable) {
        window.ApiClient.particles.emit(rect, {
          imageIds: [
            this.state.particlesSpriteFileIdFirst,
            this.state.particlesSpriteFileIdSecond,
            this.state.particlesSpriteFileIdThird,
          ].filter(s => !!s),
          particlesCountMin: parseFloat(this.state.particlesCountMin) || 0,
          particlesCountMax: parseFloat(this.state.particlesCountMax) || 0,
          particlesDurationMin: parseFloat(this.state.particlesDurationMin) || 0,
          particlesDurationMax: parseFloat(this.state.particlesDurationMax) || 0,
          particlesDirectionXMin: parseFloat(this.state.particlesDirectionXMin) || 0,
          particlesDirectionXMax: parseFloat(this.state.particlesDirectionXMax) || 0,
          particlesDirectionYMin: parseFloat(this.state.particlesDirectionYMin) || 0,
          particlesDirectionYMax: parseFloat(this.state.particlesDirectionYMax) || 0,
          particlesScaleMin: parseFloat(this.state.particlesScaleMin) || 1,
          particlesScaleMax: parseFloat(this.state.particlesScaleMax) || 1,
          particlesRotationMin: parseFloat(this.state.particlesRotationMin) || 0,
          particlesRotationMax: parseFloat(this.state.particlesRotationMax) || 0,
        });
      }

      if (this.state.soundEnable && this.state.soundFile) {
        window.ApiClient.sound.playFile(this.state.soundFile, {
          volume: this.state.soundVolume ?? 1,
          detuneMin: this.state.soundDetuneMin ?? 0,
          detuneMax: this.state.soundDetuneMax ?? 0,
          playbackMin: this.state.soundPlaybackMin ?? 1,
          playbackMax: this.state.soundPlaybackMax ?? 1,
        });
      }
    } catch (error) {

    }
  }

  // start/restart activity timer
  triggerActivity() {
    // reset timer
    clearTimeout(this.activityTimerHandle);
    clearTimeout(this.activityTimerDelayClearHandle);

    if (!this.boxElement.classList.contains("active")) {
      this.onShow();
      this.boxElement.classList.add("active");
    }

    this.activityTimerHandle = setTimeout(() => {
      this.boxElement.classList.remove("active");

      this.onHide();

      this.activityTimerDelayClearHandle = setTimeout(() => {
        this.clearSentences();
      }, this.state.behaviorClearDelay) as unknown as number;
    }, this.state.behaviorClearTimer) as unknown as number;
  }

  #stopAnimation() {
    clearTimeout(this.animationTimerHandle);
    this.animatingSentence = null;
    this.isPlayingAnimation = false;
  }

  clearSentences() {
    this.#stopAnimation();
    for (const v of this.sentenceQueue) {
      this.retiredLines.retire(v.line);
      v.element.remove();
    }
    this.sentenceQueue.length = 0;
    this.nextAnimationIndex = 0;
  }

  /** Removes one line (its text was cleared) and moves the animation on if it was playing it. */
  removeSentence(index: number) {
    const [sentence] = this.sentenceQueue.splice(index, 1);
    if (!sentence) return;
    if (sentence === this.animatingSentence) this.#stopAnimation();
    if (index < this.nextAnimationIndex) this.nextAnimationIndex--;
    this.retiredLines.retire(sentence.line);
    sentence.element.remove();
    this.tryAnimateNextSentence();
  }

  stepSentenceAnimation(sentence: SentenceState) {
    if (this.disposed) {
      return
    }
    const line = sentence.line;
    // end of the line's animation
    if (!line.pending) {
      line.finish();
      sentence.isAnimated = true;
      this.isPlayingAnimation = false;
      this.animatingSentence = null;
      this.tryAnimateNextSentence();
      return;
    }

    this.isPlayingAnimation = true;
    this.animatingSentence = sentence;

    this.triggerActivity();
    line.text[line.cursor] !== " " && this.onActivity();

    // a whole word per step when there is no per-character delay (spaces still step one by one)
    line.step(this.profanityMask, this.state.animateDelayChar === 0 ? "word" : "char");

    const delay = line.text[line.cursor] === " " ?
      this.state.animateDelayWord - this.state.animateDelayChar :
      this.state.animateDelayChar;

    this.animationTimerHandle = setTimeout(() => this.stepSentenceAnimation(sentence), delay) as unknown as number;
  }

  tryAnimateNextSentence() {
    if (this.isPlayingAnimation) {
      return;
    }

    // Use cached index for O(1) lookup instead of scanning
    while (this.nextAnimationIndex < this.sentenceQueue.length && this.sentenceQueue[this.nextAnimationIndex].isAnimated) {
      this.nextAnimationIndex++;
    }
    if (this.nextAnimationIndex < this.sentenceQueue.length) {
      this.stepSentenceAnimation(this.sentenceQueue[this.nextAnimationIndex]);
    }
  }

  enqueueSentence(event: TextEvent, topic: string) {
    const kind: CaptionKind = event.type === TextEventType.interim ? "interim" : "final";
    if (!this.state.sourceInterim && kind === "interim") {
      return;
    }

    // Pair the update with its line: segmentId, then id, then (sources with neither) the topic's open interim.
    // The AI transform publishes each synced line on two topics (same id and segment), so on "Any text source"
    // they share one line, as they always did.
    const pairTopic = topic === TextEventSource.transform_raw ? TextEventSource.transform_source : topic;
    const meta = readCaptionMeta(pairTopic, kind, event);
    const index = findCaptionLine(this.sentenceQueue.map(s => s.line), meta);
    const target = index === -1 ? undefined : this.sentenceQueue[index];
    // older revision, an interim after the final, or a late update of a line that already left the screen
    if (target ? isStaleUpdate(target.line, meta) : this.retiredLines.blocks(meta)) {
      return;
    }

    const value = typeof event.value === "string" ? event.value : "";
    // an empty text removes the line (the text field was cleared)
    if (!value) {
      if (target) {
        this.removeSentence(index);
        if (!this.sentenceQueue.length) {
          this.boxElement.classList.remove("active");
        }
      }
      return;
    }

    const animate = this.state.animateEnable;
    const update: CaptionUpdate = {
      kind,
      text: value,
      emotes: event.emotes,
      stableLength: event.stableLength,
      mode: animate ? "positions" : "names",
      meta,
    };

    let sentence = target;
    let at = index;
    if (sentence) {
      // keeps what is still correct on screen; only the rest is rendered again
      sentence.line.update(update);
    } else {
      // A new segment: the stream's open interims of earlier segments were given up upstream (nothing more
      // comes for them), so they leave now instead of lingering until the clear timer.
      const superseded = supersededCaptionLines(this.sentenceQueue.map(s => s.line), meta);
      for (let i = superseded.length - 1; i >= 0; i--) {
        this.removeSentence(superseded[i]);
      }
      this.retiredLines.revive(meta);
      const element = document.createElement("span");
      sentence = { line: new CaptionLine<Node>(DOM_CAPTIONS, element, meta), element, isAnimated: false };
      sentence.line.update(update);
      at = this.sentenceQueue.push(sentence) - 1;
      this.textElement.insertBefore(element, this.textEndElement);
    }

    if (animate) {
      if (sentence.line.pending) {
        // (re)open the line; an earlier line animates before the ones after it
        sentence.isAnimated = false;
        this.nextAnimationIndex = Math.min(this.nextAnimationIndex, at);
        this.tryAnimateNextSentence();
      } else if (sentence !== this.animatingSentence) {
        // nothing new to show (same text, or shortened): the line is complete as it is
        sentence.line.finish();
        sentence.isAnimated = true;
      }
    } else {
      sentence.line.flush(this.profanityMask);
      sentence.line.finish();
      sentence.isAnimated = true;
      this.onActivity();
    }
    this.triggerActivity();
  }

  /**
   * The /client page (re)connected to its host. After a reconnect the host may be a restarted Sigil whose segment
   * and transform ids start over (`mic:1:1`, id 0, ...): the retired ids must not block them, and the lines still
   * on screen must not pair with them (their interims will never be updated again and leave; finals stay, unpaired).
   */
  onHostSynced(reconnected: boolean) {
    if (reconnected) {
      const gone = detachCaptionLines(this.sentenceQueue.map(s => s.line));
      for (let i = gone.length - 1; i >= 0; i--) {
        this.removeSentence(gone[i]);
      }
      if (gone.length && !this.sentenceQueue.length) {
        this.boxElement.classList.remove("active");
      }
    }
    this.retiredLines.clear();
  }

  onResize() {
    this.scrollContainerElement.scrollTo({ top: this.scrollContainerElement.scrollHeight, behavior: "smooth" });
  }

  // copy state for faster access
  onStateChange() {
    if (this.disposed || !this.stateStyleElement) return;
    // detect changes
    const scene = window.ApiClient.scenes.state.activeScene;
    const el = window.ApiClient.document.fileBinder.get().elements[this.id];
    const storedState = el?.scenes[scene]?.data as Element_TextState | undefined;
    if (!storedState) return;
    if (fastEqual(storedState, this.currentState)) {
      return;
    }
    this.currentState = storedState;

    // update styles
    this.stateStyleElement.innerHTML = buildStateStyle(this.currentState);

    // rebind events
    this.bindTextEvents();

    if (this.currentState.previewMode) {
      this.boxElement.classList.add("active");
    }
    else {
      this.boxElement.classList.remove("active");
    }
  }


  bindTextEvents() {
    const main = this.state.sourceMain;

    // primary event sub (typed text is taken from the subscription below only, so "Any text source" shows it once)
    window.ApiShared.pubsub.unsubscribe(this.eventTextRef);
    this.eventTextRef = window.ApiShared.pubsub.subscribeText(main, (e, topic) => {
      if (!e?.value || topic === TextEventSource.textfield) return;
      this.enqueueSentence(e, topic ?? main);
    });

    // kb input event sub (also delivers the empty events that clear the typed line)
    window.ApiShared.pubsub.unsubscribe(this.eventTextInputRef);
    this.eventTextInputRef = window.ApiShared.pubsub.subscribeText(TextEventSource.textfield, e => {
      if (!e) return;
      if (main !== TextEventSource.textfield && !this.state.sourceInputField) return;
      this.enqueueSentence(e, TextEventSource.textfield);
    }, true);
  }

  /** Ref callback of the container: binds it once, and releases it when it unmounts or is replaced. */
  bindContainer(containerElement: HTMLDivElement | null) {
    if (this.disposed || containerElement === this.containerElement) {
      return;
    }
    this.unbindContainer();
    if (!containerElement) {
      return;
    }

    this.containerElement = containerElement;
    this.boxElement = this.containerElement.querySelector(".box") as HTMLElement;
    this.textElement = this.containerElement.querySelector(".text") as HTMLElement;
    this.textEndElement = this.containerElement.querySelector(".text-end") as HTMLSpanElement;
    this.scrollContainerElement = this.containerElement.querySelector(".scroll-container") as HTMLElement;

    // create base style ele
    this.baseStyleElement = document.createElement("style");
    this.baseStyleElement.innerHTML = elementStyle;
    this.containerElement.append(this.baseStyleElement);

    // create state style ele
    this.stateStyleElement = document.createElement("style");
    this.containerElement.append(this.stateStyleElement);

    // listen for text or container sizes
    this.textObserver = new ResizeObserver(e => this.onResize());
    this.textObserver.observe(this.textElement);
    this.textObserver.observe(this.scrollContainerElement);

    // subscribe to store updates
    this.sceneChangeEventCancelToken = subscribe(window.ApiClient.scenes.state, () => this.onStateChange());
    this.storeEventCancelToken = window.ApiClient.document.fileBinder.subscribe(data => this.onStateChange());
    this.fileUrlsUnsub = subscribe(window.ApiClient.files.fileUrls, () => {
      if (this.disposed || !this.stateStyleElement) return;
      const scene = window.ApiClient.scenes.state.activeScene;
      const el = window.ApiClient.document.fileBinder.get().elements[this.id];
      const storedState = el?.scenes[scene]?.data as Element_TextState | undefined;
      if (!storedState) return;
      this.stateStyleElement.innerHTML = buildStateStyle(storedState);
    });

    // /client pages outlive the host (see onHostSynced). On the host the phase stays idle.
    const mirror = window.ApiShared?.peer?.clientMirrorState;
    if (mirror) {
      let phase = mirror.phase;
      this.peerMirrorUnsub = subscribe(mirror, () => {
        if (mirror.phase === "synced" && phase !== "synced") {
          this.onHostSynced(phase === "reconnecting" || phase === "failed");
        }
        phase = mirror.phase;
      });
    }

    this.onStateChange();
  }

  /** Releases everything bound to the current container (its lines, observers, subscriptions and styles). */
  unbindContainer() {
    if (!this.containerElement) return;
    this.clearSentences();
    clearTimeout(this.activityTimerHandle);
    clearTimeout(this.activityTimerDelayClearHandle);
    this.textObserver?.disconnect();
    this.textObserver = undefined;
    this.sceneChangeEventCancelToken?.();
    this.storeEventCancelToken?.();
    this.fileUrlsUnsub?.();
    this.peerMirrorUnsub?.();
    this.sceneChangeEventCancelToken = this.storeEventCancelToken = this.fileUrlsUnsub = this.peerMirrorUnsub = undefined;
    window.ApiShared.pubsub.unsubscribe(this.eventTextRef);
    window.ApiShared.pubsub.unsubscribe(this.eventTextInputRef);
    this.eventTextRef = this.eventTextInputRef = undefined;
    this.baseStyleElement?.remove();
    this.stateStyleElement?.remove();
    this.baseStyleElement = this.stateStyleElement = undefined;
    this.currentState = undefined;
    this.containerElement = undefined;
  }

  dispose() {
    this.unbindContainer();
    this.disposed = true;
    clearTimeout(this.animationTimerHandle);
    clearTimeout(this.activityTimerHandle);
    clearTimeout(this.activityTimerDelayClearHandle);
  }
}
// state.previewMode
const Element_Text: FC<{ id: string }> = memo(({ id }) => {
  const controller = useRef(new TextController(id));
  const bindContainer = useCallback((ele: HTMLDivElement | null) => controller.current.bindContainer(ele), []);
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const state = useGetState(state =>
    state.elements[id]?.scenes[activeScene]?.data as Element_TextState | undefined
  );

  useEffect(() => {
    return () => controller.current.dispose();
  }, []);

  if (!state) return null;

  return <div className="container" ref={bindContainer}>
    <div className="box">
      <div className="scroll-container">
        <div className="text-container">
          <span className="text">
            {state.previewMode && <span>This is <span className="profanity">{state.textProfanityMask || '***'}</span> test! <span className="interim">This is interim test</span> </span>}
            <span className="text-end"></span>
          </span>
        </div>
      </div>
    </div>
  </div>
});
export default Element_Text;
