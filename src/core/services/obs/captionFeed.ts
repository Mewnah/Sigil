import { ObsCaptionEnvelope, TextEvent, TextEventType } from "@/types";
import { serviceSubscibeToInput, serviceSubscibeToSource } from "../../../utils";
import { OBS_State } from "./schema";

type FeedPublisher = (envelope: ObsCaptionEnvelope) => void;

export class ObsCaptionFeedService {
  private eventDisposers: (() => void)[] = [];

  constructor(
    private readonly getState: () => OBS_State,
    private readonly publish: FeedPublisher,
  ) {}

  init() {
    const state = this.getState();
    this.eventDisposers.push(
      serviceSubscibeToSource(state, "browserSource", (e) => this.processTextEvent(e)),
    );
    this.eventDisposers.push(
      serviceSubscibeToInput(state, "browserInputField", (e) => this.processTextEvent(e)),
    );
  }

  private processTextEvent(data?: TextEvent) {
    const state = this.getState();
    if (
      !state.browserCaptionsEnable ||
      !data?.value ||
      !(data.type === TextEventType.final || (data.type === TextEventType.interim && state.browserInterim))
    ) {
      return;
    }

    this.publish({
      v: 1,
      source: state.browserSource,
      type: data.type === TextEventType.final ? "final" : "interim",
      value: data.value,
      at: Date.now(),
    });
  }

  dispose() {
    this.eventDisposers.forEach((d) => d());
    this.eventDisposers = [];
  }
}

