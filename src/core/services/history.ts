import { proxy } from "valtio";

export interface HistorySnapshot {
    timestamp: number;
    description: string;
    data: string;
}

/**
 * Placeholder service: document undo/redo is implemented in the client layer
 * (`documentUndoState` / `ApiClient.document`), not here. Kept on `ApiServer` for API stability.
 */
export default class Service_History {
    store = proxy({
        past: [] as HistorySnapshot[],
        future: [] as HistorySnapshot[],
        canUndo: false,
        canRedo: false
    });

    init(_state: unknown) {
        // Intentionally empty — see class docstring.
    }

    pushSnapshot(description: string) { }
    undo() { }
    redo() { }
    restore(timestamp: number) { }
}
