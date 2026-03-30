import z from "zod";
import { ElementType } from "./client/elements/schema";
import { Services } from "./services-registry";

export type MappedGroupDictionary<Options = any> = Record<string, ([string] | [string, string] | [string, string, Options])[]>

export interface IServiceInterface {
  init(): void;
}

export const enum ServiceNetworkState {
  disconnected = "disconnected",
  connecting = "connecting",
  connected = "connected",
  error = "error"
}

export enum TextEventType {
  final,
  interim,
}

const TextEventTypeSchema = z.nativeEnum(TextEventType);

export type BaseEvent<Data = unknown> = {
  topic: string,
  data?: Data
}

export type PartialWithRequired<T, K extends keyof T> = Pick<T, K> & Partial<T>;

export const TextEventSchema = z.object({
  type: TextEventTypeSchema,
  value: z.string().default(""),
  emotes: z.record(z.number().or(z.string()), z.string()).default({}),
  textFieldType: z.enum(["textField", "twitchChat", "kickChat"]).nullish(),
  id: z.number().optional()
});

export type TextEvent = z.infer<typeof TextEventSchema>;

export enum TextEventSource {
  any = "text",
  textfield = "text.textfield",
  stt = "text.stt",
  translation = "text.translation",
  transform = "text.transform",
  transform_source = "text.transform_source",
  transform_raw = "text.transform_raw",
}

export const TextEventSourceSchema = z.nativeEnum(TextEventSource);

export type InspectorTabPath = {
  tab: Services | ElementType | "settings" | "integrations" | "scenes" | "files" | "fonts" | "project" | "elements",
  value?: string
}
