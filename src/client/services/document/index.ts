import { IServiceInterface } from "@/types";
import i18n from "i18next";
import { open, save } from "@tauri-apps/plugin-dialog";
import { BaseDirectory } from "@tauri-apps/api/path";
import { exists, mkdir, readFile, writeFile } from "@tauri-apps/plugin-fs";
import { Binder, bind } from "immer-yjs";
import debounce from "lodash/debounce";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import * as Y from "yjs";
import { ElementType } from "../../elements/schema";
import { migrateLegacyJsonDocument } from "@/client/migrate/legacyDocument";
import { DocumentSchema, DocumentState } from "../../schema";

/** Reactive undo/redo availability for the template document (Yjs UndoManager). */
export const documentUndoState = proxy({
  canUndo: false,
  canRedo: false,
});

class Service_Document implements IServiceInterface {
  #file: Y.Doc = new Y.Doc();
  fileBinder!: Binder<DocumentState>;
  #undoManager!: Y.UndoManager;

  get template() {
    return this.#file?.getMap("template");
  }

  get fileArray() {
    return this.#file.getArray<Uint8Array>("files");
  }

  get file() {
    return this.#file;
  }

  createNewState() {
    this.patch(state => {
      const newState = DocumentSchema.parse({});
      let k: keyof DocumentState;
      for (k in newState)
        this.#patchField(state, newState, k);
    });
    const canvas = this.fileBinder.get().canvas;
    // add default text element
    const eleWidth = canvas.w - 100;
    window.ApiClient.elements.addElement(ElementType.text, "main", {
      w: eleWidth,
      h: 65,
      x: (canvas.w - eleWidth) / 2,
      y: (canvas.h - 65) / 2,
      r: 0
    });
  }

  // i hate this
  #patchField<Key extends keyof DocumentState>(og: DocumentState, patch: DocumentState, key: Key) {
    og[key] = patch[key];
  }

  patchState(immerState: DocumentState, newState: DocumentState) {
    // trigger immer-yjs generator
    let k: keyof DocumentState;
    for (k in newState)
      this.#patchField(immerState, newState, k);
    // remove fields
    for (let k in immerState) {
      if (!(k in newState))
        delete immerState[k as keyof DocumentState]
    }
  }

  /** Replaces the Yjs `files` array (embedded fonts/images) used with `filesMeta`. */
  #replaceYFileBinaries(chunks: Uint8Array[]) {
    const dest = this.#file.getArray<Uint8Array>("files");
    this.#file.transact(() => {
      if (dest.length > 0) {
        dest.delete(0, dest.length);
      }
      for (const u of chunks) {
        dest.push([new Uint8Array(u)]);
      }
    });
  }

  #syncDocumentUndoState = () => {
    documentUndoState.canUndo = this.#undoManager.canUndo();
    documentUndoState.canRedo = this.#undoManager.canRedo();
  };

  #initUndoManager() {
    const templateMap = this.#file.getMap("template");
    this.#undoManager = new Y.UndoManager(templateMap);
    const sync = () => this.#syncDocumentUndoState();
    this.#undoManager.on("stack-item-added", sync);
    this.#undoManager.on("stack-item-popped", sync);
    this.#undoManager.on("stack-cleared", sync);
    sync();
  }

  undo() {
    this.#undoManager.undo();
    this.#syncDocumentUndoState();
  }

  redo() {
    this.#undoManager.redo();
    this.#syncDocumentUndoState();
  }

  async init() {
    this.#file.getArray<Uint8Array>("files");
    this.fileBinder = bind<DocumentState>(this.#file.getMap("template"));
    this.#initUndoManager();

    if (window.Config.isClient()) {
      // Initial Yjs sync is applied during peer.startClient() on this same Doc, often *before*
      // we attach a listener — `once("update")` would then hang forever and the mirror never finishes init.
      const template = this.#file.getMap("template");
      if (template.size === 0) {
        await new Promise<void>((resolve) => {
          this.#file.once("update", () => resolve());
        });
      }
      this.#undoManager.clear(true, true);
      this.#syncDocumentUndoState();
      return;
    }

    const loadState = await this.loadDocument();
    if (loadState) {
      Y.applyUpdate(this.#file, loadState);
      this.patch((state) => {
        const patchState = DocumentSchema.safeParse(state);
        if (patchState.success) {
          this.patchState(state, patchState.data);
        }
        else {
          toast.error(i18n.t("toasts.invalid_template"));
          this.createNewState();
        }
      });
    }
    else {
      this.createNewState();
    }
    this.#undoManager.clear(true, true);
    this.#syncDocumentUndoState();
    this.saveDocument();
    this.#file.on("afterTransaction", (e) => {
      this.saveDocument();
    });
  }

  async importDocument() {
    if (window.Config.isClient()) {
      toast.error(i18n.t("toasts.import_host_only"));
      return;
    }
    const path = await open({
      filters: [
        {
          name: "Template (.sigiltmp, .cursestmp, JSON)",
          extensions: ["sigiltmp", "cursestmp", "json", "sigil"],
        },
      ],
    });
    if (!path || Array.isArray(path)) return;

    const data = await readFile(path);

    const tryParseJson = (): unknown | null => {
      let i = 0;
      if (data.length >= 3 && data[0] === 0xef && data[1] === 0xbb && data[2] === 0xbf) i = 3;
      if (i >= data.length || data[i] !== 0x7b /* { */) return null;
      try {
        const text = new TextDecoder("utf-8").decode(data.subarray(i));
        return JSON.parse(text) as unknown;
      } catch {
        return null;
      }
    };

    const json = tryParseJson();
    if (json !== null && typeof json === "object") {
      const migrated = migrateLegacyJsonDocument(json);
      if (!migrated.ok) {
        toast.error(migrated.error);
        return;
      }
      this.patch((state) => {
        Object.assign(state, migrated.doc);
      });
      this.#undoManager.clear(true, true);
      this.#syncDocumentUndoState();
      await this.#saveDocumentNative(this.#file);
      if (migrated.notes.length > 0) {
        toast.info(i18n.t("project.migrate_toast", { details: migrated.notes.join(" ") }));
      } else {
        toast.success(i18n.t("project.toast_imported"));
      }
      return;
    }

    const tempDoc = new Y.Doc();
    try {
      Y.applyUpdate(tempDoc, data);
    } catch {
      toast.error(i18n.t("toasts.invalid_template_file"));
      return;
    }
    const tempBinder = bind<DocumentState>(tempDoc.getMap("template"));
    const snapshot = tempBinder.get();

    const yFiles = tempDoc.getArray<Uint8Array>("files");
    const fileBinaries: Uint8Array[] = [];
    for (let i = 0; i < yFiles.length; i++) {
      const chunk = yFiles.get(i);
      fileBinaries.push(chunk instanceof Uint8Array ? new Uint8Array(chunk) : new Uint8Array());
    }

    const parsed = DocumentSchema.safeParse(snapshot);
    if (parsed.success) {
      this.#replaceYFileBinaries(fileBinaries);
      this.patch((state) => {
        this.patchState(state, parsed.data);
      });
      this.#undoManager.clear(true, true);
      this.#syncDocumentUndoState();
      await this.#saveDocumentNative(this.#file);
      toast.success(i18n.t("project.toast_imported"));
      return;
    }

    const migrated = migrateLegacyJsonDocument(snapshot, fileBinaries);
    if (!migrated.ok) {
      toast.error(migrated.error);
      return;
    }
    this.#replaceYFileBinaries(migrated.alignedFileBinaries ?? []);
    this.patch((state) => {
      this.patchState(state, migrated.doc);
    });
    this.#undoManager.clear(true, true);
    this.#syncDocumentUndoState();
    await this.#saveDocumentNative(this.#file);
    if (migrated.notes.length > 0) {
      toast.info(i18n.t("project.migrate_toast", { details: migrated.notes.join(" ") }));
    } else {
      toast.success(i18n.t("project.toast_imported"));
    }
  }

  /** Replace the working template with defaults (new canvas + one text element). Saves immediately. */
  async resetTemplate(): Promise<void> {
    if (window.Config.isClient()) {
      toast.error(i18n.t("toasts.reset_host_only"));
      return;
    }
    this.createNewState();
    this.#undoManager.clear(true, true);
    this.#syncDocumentUndoState();
    await this.#saveDocumentNative(this.#file);
  }
  async exportDocument(authorName: string) {
    // clone doc
    const tempDoc = new Y.Doc();
    const encodedUpdate = Y.encodeStateAsUpdate(this.#file);
    Y.applyUpdate(tempDoc, encodedUpdate);
    // apply author name to temp
    tempDoc.getMap("template").set("author", authorName);

    const tempEncodedUpdate = Y.encodeStateAsUpdate(tempDoc);
    const path = await save({
      filters: [
        {
          name: "Sigil template",
          extensions: ["sigiltmp"],
        },
      ],
    });
    if (path) try {
      await writeFile(path, tempEncodedUpdate);
      // write author to original doc on success
      this.fileBinder.update(a => { a.author = authorName });
    } catch (error) {

    }
  }

  async loadDocument(): Promise<Uint8Array | undefined> {
    if (window.Config.isClient()) {
      return;
    }

    const bExists = await exists("user/template", {
      baseDir: BaseDirectory.AppData,
    });
    if (bExists) try {
      const data = await readFile("user/template", {
        baseDir: BaseDirectory.AppData,
      });
      return data;
    } catch (error) {
      toast("Error loading template", { type: "error" });
    }
  }

  async #saveDocumentNative(doc: Y.Doc) {
    const bExists = await exists("user", { baseDir: BaseDirectory.AppData });
    if (!bExists)
      await mkdir("user", { baseDir: BaseDirectory.AppData, recursive: true });
    const data = Y.encodeStateAsUpdate(doc);
    await writeFile("user/template", data, { baseDir: BaseDirectory.AppData });
  }

  saveDocument = debounce(() => {
    if (window.Config.isClient())
      return;
    this.#saveDocumentNative(this.#file);
  }, 2000);

  patch(patchFn: (state: DocumentState) => void) {
    this.file.transact((_) => {
      this.fileBinder.update(patchFn);
    });
  }
}

export default Service_Document;
