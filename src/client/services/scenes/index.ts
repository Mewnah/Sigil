import { IServiceInterface } from "@/types";
import { nanoid }            from "nanoid";
import { proxy } from "valtio";
import { SceneStateSchema } from "./schema";

class Service_Scenes implements IServiceInterface {
  state = proxy({
    activeScene: 'main'
  });

  get scenes() {
    return window.ApiClient.document.fileBinder.get().scenes;
  }

  init(): void {
    if (window.Config.isClient() && window.Config.clientInitialState?.scene) {
      this.state.activeScene = window.Config.clientInitialState?.scene ?? "main";
    }
    else 
      this.state.activeScene = window.ApiClient.document.fileBinder.get().activeScene;
    window.ApiShared.pubsub.subscribe("scenes:change", e => {
      if (typeof e === "string" && e in this.scenes) {
        this.state.activeScene = e;
      }
    });
  }

  addScene() {
    window.ApiClient.document.patch((state) => {
      let id = nanoid();
      while (id in state.scenes)
        id = nanoid();
      state.scenes[id] = SceneStateSchema.parse({id});
    });
  }

  duplicateScene(ogId: string) {
    window.ApiClient.document.patch((state) => {
      if (!(ogId in state.scenes))
        return;

      let id = nanoid();
      while (id in state.scenes)
        id = nanoid();

      // clone scene
      state.scenes[id] = {...state.scenes[ogId]};
      state.scenes[id].id = id;
      state.scenes[id].name += ' (Copy)';

      // clone scene data in elements
      for (const elId in state.elements) {
        if (ogId in state.elements[elId].scenes) {
          state.elements[elId].scenes[id] = {...state.elements[elId].scenes[ogId]};
        }
      }
    });
  }

  setActive(id: string): void {
    window.ApiShared.pubsub.publish("scenes:change", id);
  }

  deleteScene(id: string) {
    const wasActive = this.state.activeScene === id;
    if (wasActive)
      this.state.activeScene = "main";

    window.ApiClient.document.patch((state) => {
      if (id in state.scenes)
        delete state.scenes[id];
      for (const elementKey in state.elements) {
        if (id in state.elements[elementKey].scenes) {
          delete state.elements[elementKey].scenes[id];
        }
      }
    });

    // The switch above is local: a mirror still showing the deleted scene would draw nothing.
    if (wasActive)
      window.ApiShared.pubsub.publish("scenes:change", "main");
  }
}
export default Service_Scenes;
