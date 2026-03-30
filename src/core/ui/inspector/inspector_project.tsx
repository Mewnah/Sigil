import { FC, memo, useEffect, useState } from "react";
import {
  RiFileAddLine,
  RiFileCopyLine,
  RiFolderOpenLine,
  RiLayoutMasonryFill,
  RiStackFill,
} from "react-icons/ri";
import { useGetState } from "@/client";
import { useTranslation } from "react-i18next";
import Inspector from "./components";
import { InputBaseText } from "./components/input";
import { resetTemplate } from "../projectFileActions";

const Inspector_Project: FC = memo(() => {
  const { t } = useTranslation();
  const author = useGetState((s) => s.author);
  const canvas = useGetState((s) => s.canvas);
  const scenes = useGetState((s) => s.scenes);
  const elementsIds = useGetState((s) => s.elementsIds);

  const [exportName, setExportName] = useState("");
  useEffect(() => {
    setExportName(author ?? "");
  }, [author]);

  const sceneCount = scenes ? Object.keys(scenes).length : 0;
  const elementCount = elementsIds?.length ?? 0;

  const isClient = window.Config.isClient();
  const isApp = window.Config.isApp();

  const goToCanvas = () => {
    window.ApiServer.changeTab({ tab: "scenes" });
  };

  return (
    <Inspector.Body>
      <Inspector.Header>
        <RiStackFill />
        {t("project.title")}
      </Inspector.Header>
      <Inspector.Content>
        <Inspector.SubHeader>{t("project.section_template")}</Inspector.SubHeader>
        <Inspector.Description>
          <span>{t("project.template_live_copy")}</span>
        </Inspector.Description>
        <p className="text-xs text-base-content/70 leading-relaxed">
          {isApp && !isClient ? t("project.template_autosave_desktop") : t("project.template_autosave_client")}
        </p>

        <button
          type="button"
          className="btn btn-sm btn-error btn-outline gap-2 justify-start mt-2"
          onClick={() => void resetTemplate()}
        >
          <RiFileAddLine className="text-lg" />
          {t("project.btn_reset_template")}
        </button>
        <button
          type="button"
          className="btn btn-sm btn-outline gap-2 justify-start"
          onClick={() => void window.ApiClient.document.importDocument()}
        >
          <RiFolderOpenLine className="text-lg" />
          {t("project.btn_import_template")}
        </button>
        <div className="flex flex-col gap-2 pt-1">
          <span className="text-xs text-base-content/60">{t("project.field_export_author")}</span>
          <InputBaseText
            fieldWidth={false}
            className="w-full"
            value={exportName}
            onChange={(e) => setExportName(e.target.value)}
            placeholder={t("project.field_export_author")}
          />
          <button
            type="button"
            className="btn btn-sm btn-outline gap-2 justify-start"
            onClick={() => {
              const name = exportName.trim();
              if (name) void window.ApiClient.document.exportDocument(name);
            }}
          >
            <RiFileCopyLine />
            {t("project.btn_export_template")}
          </button>
        </div>

        <Inspector.SubHeader>{t("project.section_summary")}</Inspector.SubHeader>
        <ul className="text-sm space-y-1 text-base-content/80">
          <li>
            {t("project.field_canvas")}: {canvas?.w ?? "—"} × {canvas?.h ?? "—"}
          </li>
          <li>
            {t("project.field_scenes")}: {sceneCount}
          </li>
          <li>
            {t("project.field_elements")}: {elementCount}
          </li>
          <li>
            {t("project.field_author")}: {author?.trim() ? author : "—"}
          </li>
        </ul>
        <button type="button" className="btn btn-sm btn-primary gap-2 w-full mt-2" onClick={goToCanvas}>
          <RiLayoutMasonryFill />
          {t("project.btn_edit_canvas")}
        </button>
      </Inspector.Content>
    </Inspector.Body>
  );
});

export default Inspector_Project;
