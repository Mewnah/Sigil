import { TextEvent } from "@/types";

export interface ITransformService {
    start(data: any): void;
    stop(): void;
    transform(id: number, e: TextEvent, history: { role: "user" | "assistant", content: string }[]): void;
}

export interface ITransformReceiver {
    onStart(): void;
    onStop(error?: string): void;
    onTransform(id: number, e: TextEvent, value: string): void;
}

export interface ITransformServiceConstructor {
    new(receiver: ITransformReceiver): ITransformService;
}
