import { ITransformEngine, TransformRequest } from "../types";

/** Debug backend: upper-cases the line, no network. */
export class Transform_DummyService implements ITransformEngine {
    async start(): Promise<void> {}

    stop(): void {}

    async transform(request: TransformRequest): Promise<string> {
        return request.text.toUpperCase() + " (AI)";
    }
}
