import { ITransformReceiver, ITransformService } from "../types";
import { TextEvent } from "@/types";

export class Transform_DummyService implements ITransformService {
    constructor(private receiver: ITransformReceiver) { }

    start(data: any): void {
        this.receiver.onStart();
    }

    stop(): void {
        this.receiver.onStop();
    }

    transform(id: number, e: TextEvent, history: any[]): void {
        const transformed = e.value.toUpperCase() + " (AI)";
        this.receiver.onTransform(id, e, transformed);
    }
}
