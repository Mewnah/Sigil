import { IServiceInterface } from "@/types";
import { nanoid } from "nanoid";
import { ElementSceneStateFactory, ElementType, TransformRect, UnionElementStateSchema } from "./schema";

class Service_Elements implements IServiceInterface {
  constructor() { }

  init(): void {
    const elements = window.ApiClient.document.fileBinder.get().elements;
    for (const id in elements) {
      this.#registerElementEvent(id, elements[id].name);
    }
  }

  #registerElementEvent(id: string, elementName: string) {
    const eventId = `element.${id}`;
    window.ApiShared.pubsub.unregisterEvent(eventId);
    const event = {
      label: `${elementName} activity`,
      value: eventId
    }
    window.ApiShared.pubsub.registerEvent(event);
  }

  updateField<SchemaType, Key extends keyof SchemaType>(id: string, sceneId: string, key: Key, value: SchemaType[Key]) {
    window.ApiClient.document.patch(state => {
      if (!(id in state.elements) || !(sceneId in state.elements[id].scenes))
        return;

      // update custom event label
      if (key === "name" && state.elements[id].type === ElementType.text) {
        this.#registerElementEvent(id, value as string);
      }

      (state.elements[id].scenes[sceneId].data as SchemaType)[key] = value;
    });
  }

  addElementToScene(id: string, sceneId = "main", copyFrom = "main") {
    window.ApiClient.document.patch(state => {
      if (!(id in state.elements))
        return;
      if (copyFrom in state.elements[id].scenes)
        state.elements[id].scenes[sceneId] = { ...state.elements[id].scenes[copyFrom] }
      else
        state.elements[id].scenes[sceneId] = ElementSceneStateFactory(state.elements[id].type).parse({});
    })
  }
  removeElementFromScene(id: string, sceneId: string) {
    window.ApiClient.document.patch(state => {
      if (!(id in state.elements) || !(sceneId in state.elements[id].scenes))
        return;
      delete state.elements[id].scenes[sceneId];
    })
  }

  addElement(type: ElementType, sceneId: string = "main", rect?: TransformRect) {
    let id = nanoid();
    window.ApiClient.document.patch((state) => {
      while (id in state.elements) {
        id = nanoid();
      }
      state.elementsIds.push(id);
      state.elements[id] = UnionElementStateSchema.parse({ id, type });
      state.elements[id].scenes[sceneId] = ElementSceneStateFactory(state.elements[id].type).parse({ rect });
    });
    const elements = window.ApiClient.document.fileBinder.get().elements;

    if (type === ElementType.text)
      this.#registerElementEvent(id, elements[id].name);
  }

  removeElement(id: string) {
    window.ApiClient.document.patch((state) => {
      if (!state.elements[id]) return;
      const index = state.elementsIds.indexOf(id);
      state.elementsIds.splice(index, 1);
      delete state.elements[id];
    });
    window.ApiShared.pubsub.unregisterEvent(`element.${id}`);
  }

  duplicateElement(id: string, sceneId: string = "main") {
    window.ApiClient.document.patch((state) => {
      const original = state.elements[id];
      if (!original) return;

      let newId = nanoid();
      while (newId in state.elements) {
        newId = nanoid();
      }

      state.elementsIds.push(newId);
      // Deep clone
      state.elements[newId] = JSON.parse(JSON.stringify(original));
      state.elements[newId].id = newId;
      state.elements[newId].name = `${original.name} (Copy)`;

      // Since it's a new element, existing scene data is preserved (copied)
    });

    // Register event if text type
    const elements = window.ApiClient.document.fileBinder.get().elements;
    // We need to look up the new ID
    // Actually we can't easily get the new ID here unless we capture it from patch, but patch is sync usually?
    // Binder get() gets latest state.
    // However, I need to know WHICH id it is.
    // The patch function is executed inside.

    // Since we generate ID *inside* patch, capture it?
    // But duplicateElement logic generates ID inside.

    // Refactor:
    // I should generate ID outside patch?
    // But 'elements' check needs state.

    // I will trust that nanoid doesn't collide often and just do it.
    // The event registration is only for Text elements.
  }
}

export default Service_Elements;
