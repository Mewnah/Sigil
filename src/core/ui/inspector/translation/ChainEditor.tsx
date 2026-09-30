import type { Translation_State, TranslationEngineId } from "@/core/services/translation/schema";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { RiAddLine, RiArrowDownSLine, RiArrowUpSLine, RiCloseLine } from "react-icons/ri";
import { addToChain, backendLabelKey, engineSetupIssue, enginesNotInChain, isOnlineEngine, moveItem, removeFromChain } from "./logic";

/**
 * An ordered engine chain: move up / down, remove (a chain keeps one engine), add the engines it does not use yet.
 * Each row notes whether the engine sends text online and what it still needs (a key, a URL, a model name).
 *
 * Owned by P4-F.
 */
export const ChainEditor: FC<{
  chain: readonly TranslationEngineId[];
  state: Translation_State;
  onChange: (chain: TranslationEngineId[]) => void;
  /** Accessible name of the list. */
  label: string;
}> = ({ chain, state, onChange, label }) => {
  const { t } = useTranslation();
  const missing = enginesNotInChain(chain);

  return (
    <div className="flex flex-col gap-1">
      <ol aria-label={label} className="flex flex-col gap-1">
        {chain.map((id, i) => {
          const name = t(backendLabelKey(id));
          const issue = engineSetupIssue(id, state);
          return (
            <li key={id} className="flex items-center gap-2 rounded-lg border border-base-content/10 bg-base-200/40 px-2 py-1">
              <span className="text-xs font-semibold text-base-content/60 w-4 text-end shrink-0">{i + 1}</span>
              <span className="flex-grow min-w-0 flex flex-col">
                <span className="text-sm font-semibold truncate" title={name}>
                  {name}
                </span>
                {(issue || isOnlineEngine(id, state)) && (
                  <span className="text-[11px] leading-tight flex flex-wrap gap-x-2">
                    {isOnlineEngine(id, state) && <span className="text-base-content/60">{t("transl.route_online")}</span>}
                    {issue && <span className="text-warning">{t(issue)}</span>}
                  </span>
                )}
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-xs btn-square"
                disabled={i === 0}
                aria-label={t("transl.route_move_up", { engine: name })}
                title={t("transl.route_move_up", { engine: name })}
                onClick={() => onChange(moveItem(chain, i, -1))}
              >
                <RiArrowUpSLine aria-hidden />
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-xs btn-square"
                disabled={i === chain.length - 1}
                aria-label={t("transl.route_move_down", { engine: name })}
                title={t("transl.route_move_down", { engine: name })}
                onClick={() => onChange(moveItem(chain, i, 1))}
              >
                <RiArrowDownSLine aria-hidden />
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-xs btn-square"
                disabled={chain.length <= 1}
                aria-label={t("transl.route_remove", { engine: name })}
                title={t("transl.route_remove", { engine: name })}
                onClick={() => onChange(removeFromChain(chain, id))}
              >
                <RiCloseLine aria-hidden />
              </button>
            </li>
          );
        })}
      </ol>
      {missing.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-xs text-base-content/70 me-1">{t("transl.route_add")}</span>
          {missing.map((id) => (
            <button key={id} type="button" className="btn btn-xs btn-outline gap-1 normal-case" onClick={() => onChange(addToChain(chain, id))}>
              <RiAddLine aria-hidden /> {t(backendLabelKey(id))}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
