import * as Y from "yjs";
import { proxy } from "valtio";
import { IServiceInterface } from "@/types";
import { GetArrayDiff } from "@/utils";
import { createDocumentFile, FileState, FileType } from "./schema";
import { fileOpen } from "browser-fs-access";
import { nanoid } from "nanoid";
import { toast } from "react-toastify";
import { save, open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";

export interface GoogleFont {
  category: string
  family: string
  files: {
    [fontType: string]: string // file url
  }
  kind: string[]
  lastModified: string
  subsets: string[]
  variants: string[]
  version: string
}
type FontMeta = {
  family: string,
  weight: string,
  style: string
}

type SaveFile = Omit<FileState, "id" | "size">;

class Service_Files implements IServiceInterface {
  private fontsList: GoogleFont[] = [];
  #filesMetaUnobserve: (() => void) | null = null;
  /** FontFace objects registered from embedded files, so a reconcile can unregister them. */
  #loadedFonts: FontFace[] = [];
  /** Bumped by every reconcile so FontFace loads started for a previous template are dropped. */
  #reconcileGeneration = 0;
  public readonly fileUrls = proxy<{ [id: string]: string }>({});
  public readonly ui = proxy<{ fontFamilies: string[] }>({
    fontFamilies: [],
  });

  async init() {
    this.#bindFilesMetaObserver();
    window.ApiClient.document.template?.observe((event) => {
      if (event.changes.keys.has("filesMeta")) {
        this.reconcileEmbeddedAssets();
      }
    });
    this.#updateTrackedBlobs();
  }

  /** Rebind blob URLs and @font-face entries after template import replaces `filesMeta`. */
  reconcileEmbeddedAssets() {
    this.#reconcileGeneration++;
    this.#bindFilesMetaObserver();
    for (const id of Object.keys(this.fileUrls)) {
      URL.revokeObjectURL(this.fileUrls[id]);
      delete this.fileUrls[id];
    }
    for (const font of this.#loadedFonts) {
      document.fonts.delete(font);
    }
    this.#loadedFonts = [];
    this.ui.fontFamilies = [];
    this.#updateTrackedBlobs();
  }

  #bindFilesMetaObserver() {
    this.#filesMetaUnobserve?.();
    this.#filesMetaUnobserve = null;
    const yArr = this.#yFilesMeta;
    if (!yArr?.observe) return;
    const onChange = () => this.#updateTrackedBlobs();
    yArr.observe(onChange);
    this.#filesMetaUnobserve = () => yArr.unobserve(onChange);
  }

  private fontsLoaded = false;

  async loadFonts() {
    try {
      const resp = await fetch(`https://www.googleapis.com/webfonts/v1/webfonts?key=${import.meta.env.VITE_GOOGLE_FONTS_API_KEY}`);
      const data = await resp.json() as { items: GoogleFont[] };
      this.fontsList = data.items;
      this.fontsLoaded = true;
    } catch (error) { }
  }

  public async installFont(fontName: string) {
    // load font only when necessary
    if (!this.fontsLoaded)
      await this.loadFonts();

    if (!this.fontsLoaded) {
      return
    }
    const font = this.fontsList.find(f => f.family === fontName);
    if (!font)
      return;
    const variantList = font.variants.filter(variant => !variant.includes("italic"))
    const buffersPromise = Promise.all(variantList.map(fileKey => fetch(font.files[fileKey].replace("http://", "https://"))))
      .then(resp => Promise.all(resp.map(r => r.arrayBuffer())))
      .then(buffers => {
        const variants = variantList.map(variant => ({
          family: font.family,
          weight: (variant === "regular" || variant === "italic") ? "400" : variant.replace("italic", ""),
          style: variant.includes("italic") ? "italic" : "normal",
        }));
        this.installFonts(variants, buffers.map(b => new Uint8Array(b)));
      });
    toast.promise(buffersPromise, {
      pending: "Installing font",
      success: `Installed ${fontName}`,
      error: `Error installing ${fontName}`
    });
  }

  installFonts(fonts: FontMeta[], files: Uint8Array[]) {
    const meta: SaveFile[] = fonts.map(f => ({
      type: "font/ttf",
      name: f.family,
      meta: f
    }));
    this.#saveFiles(meta, files);
  }

  getFileUrl(fileId: string): string | undefined {
    return this.fileUrls[fileId];
  }
  getFileData(fileId: string): FileState | undefined {
    return this.#filesMeta.find((file) => file.id === fileId);
  }
  get #yFilesBinary() {
    return window.ApiClient.document.fileArray;
  }
  get #yFilesMeta(): Y.Array<Y.Map<unknown>> | undefined {
    return window.ApiClient.document.template?.get("filesMeta") as Y.Array<Y.Map<unknown>> | undefined;
  }
  get #filesMeta(): FileState[] {
    return this.#yFilesMeta?.toJSON() as FileState[] ?? [];
  }

  #requireYFilesMeta(): Y.Array<Y.Map<unknown>> {
    const yArr = this.#yFilesMeta;
    if (!yArr) {
      throw new Error("Document filesMeta is not initialized");
    }
    return yArr;
  }
  getFileBuffer(fileId: string) {
    const i = this.#filesMeta.findIndex(file => file.id === fileId);
    if (i < 0)
      return undefined;
    return this.#yFilesBinary.get(i);
  }

  #updateTrackedBlobs() {
    const filesMeta = this.#filesMeta;

    const { add, remove } = GetArrayDiff(
      Object.keys(this.fileUrls),
      filesMeta.map((f) => f.id)
    );

    for (let id of add) {
      const metaIndex = filesMeta.findIndex((m) => m.id === id)!;
      const meta = filesMeta[metaIndex]!;
      const rawFile = window.ApiClient.document.fileArray.get(metaIndex);
      const blob = new Blob([rawFile as any], { type: meta.type });
      this.fileUrls[id] = URL.createObjectURL(blob);

      if (meta.type.startsWith("font/")) {
        if (!meta.meta)
          continue;
        const fontMeta: FontMeta = meta.meta as FontMeta;

        if (!this.ui.fontFamilies.includes(fontMeta.family))
          this.ui.fontFamilies.push(fontMeta.family);

        const generation = this.#reconcileGeneration;
        new FontFace(fontMeta.family, rawFile as any, fontMeta).load().then(font => {
          // A reconcile (template import) since this load started makes the font stale.
          if (generation !== this.#reconcileGeneration) return;
          document.fonts.add(font);
          this.#loadedFonts.push(font);
        });
      }
    }

    for (let id of remove) {
      import.meta.env.DEV;
      URL.revokeObjectURL(this.fileUrls[id]);
      delete this.fileUrls[id];
    }
  }

  async addFile(type?: FileType) {
    const filters = {
      font: {
        name: "Fonts",
        extensions: ["ttf", "otf", "woff"],
      },
      image: {
        name: "Images",
        extensions: ["jpeg", "jpg", "png", "gif", "webp"],
      },
      audio: {
        name: "Sounds",
        extensions: ["wav", "mp3"],
      },
      video: {
        name: "Videos",
        extensions: [],
      }
    };

    const path = await open({
      filters: type ? [filters[type]] : [],
    });

    if (!path || Array.isArray(path))
      return;

    const splitPath = path.split(".");
    const extension = splitPath.pop();
    const fileName = splitPath.join(".").split("\\").pop() || "new file"

    if (!extension)
      return;

    const data = await readFile(path);

    let fileType = `${type}/${extension}`;
    if (type === "font") {
      this.installFonts([{
        family: fileName,
        weight: "900",
        style: "normal"
      }], [data]);
      return;
    }

    try {

      return this.#saveFiles([{ type: fileType, name: fileName, meta: {} }], [data]);
    } catch (error) {
      console.error(error);
    }
  }

  #saveFiles(filesMeta: SaveFile[], files: Uint8Array[]) {
    const ids: string[] = filesMeta.map((_) => nanoid());
    window.ApiClient.document.file.transact(() => {
      const metaList = this.#requireYFilesMeta();

      const meta = filesMeta.map((fileMeta, i) =>
        createDocumentFile({
          ...fileMeta,
          id: ids[i],
          size: files[i].length,
        })
      );
      metaList.push(meta);
      window.ApiClient.document.fileArray.push(files);
    });
    return ids;
  }

  removeFile(fileId: string) {
    if (!this.getFileData(fileId)) return;

    window.ApiClient.document.patch((state) => {
      const fileIndex = state.filesMeta.findIndex((f) => f.id === fileId);
      const yFilesMeta = this.#requireYFilesMeta();
      for (let j = fileIndex + 1; j < state.filesMeta.length; j++)
        yFilesMeta.get(j).set("index", j - 1);
      state.filesMeta.splice(fileIndex, 1);
      window.ApiClient.document.fileArray.delete(fileIndex);
    });
  }

  uninstallFont(family: string) {
    const ixs: number[] = [];
    this.#filesMeta.forEach((file, i) => {
      if (file.type.startsWith('font/') && file.meta?.family === family)
        ixs.push(i);
    });

    this.ui.fontFamilies = this.ui.fontFamilies.filter(f => f !== family);

    ixs.length && window.ApiClient.document.file.transact(() => {
      const yFilesMeta = this.#requireYFilesMeta();
      yFilesMeta.delete(ixs[0], ixs.length);
      this.#yFilesBinary.delete(ixs[0], ixs.length);
    });

    // find all font/* files with target family
  }
}

export default Service_Files;
