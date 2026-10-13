import { FC, memo, useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";
import type { Components } from "react-markdown";
import { RiBookReadLine, RiCloseLine } from "react-icons/ri";
import { Trans, useTranslation } from "react-i18next";

/** Manual chapters + home (README). Reference docs open via in-body links only. */
export const MANUAL_PAGES: { labelKey: string; file: string }[] = [
  { labelKey: "manual.nav_home", file: "README.md" },
  { labelKey: "manual.nav_overview", file: "00-overview.md" },
  { labelKey: "manual.nav_audio", file: "01-audio.md" },
  { labelKey: "manual.nav_studio", file: "02-studio.md" },
  { labelKey: "manual.nav_integrations", file: "03-integrations.md" },
  { labelKey: "manual.nav_system", file: "04-system.md" },
];

/** Current viewer location: an in-app page key plus an optional heading anchor. */
interface ManualLocation {
  path: string;
  hash?: string;
}

function stripHash(name: string): ManualLocation {
  const i = name.indexOf("#");
  if (i === -1) return { path: name };
  return { path: name.slice(0, i), hash: name.slice(i + 1) };
}

function manualFetchUrl(p: string): string {
  if (p === "repo-README.md") return "/repo-README.md";
  if (p === "ARCHITECTURE.md") return "/ARCHITECTURE.md";
  if (p.startsWith("docs/")) {
    return "/" + p.split("/").map((seg) => encodeURIComponent(seg)).join("/");
  }
  const segments = p.split("/").map((seg) => encodeURIComponent(seg)).join("/");
  return `/manual/${segments}`;
}

/** Repo-relative path of an in-app page key (manual chapters live under docs/manual/). */
function pageToRepoPath(key: string): string {
  if (key === "repo-README.md") return "README.md";
  if (key === "ARCHITECTURE.md") return "ARCHITECTURE.md";
  if (key.startsWith("docs/")) return key;
  return `docs/manual/${key}`;
}

/** Map a repo-relative markdown path back to an in-app page key, or null when it is not served. */
function repoPathToPage(path: string): string | null {
  if (!path.endsWith(".md")) return null;
  if (path === "README.md") return "repo-README.md";
  if (path === "ARCHITECTURE.md") return "ARCHITECTURE.md";
  if (path.startsWith("docs/manual/")) {
    // Nested pages (e.g. assets/SHOTLIST.md) are mirrored verbatim by scripts/sync-manual.mjs.
    return path.slice("docs/manual/".length) || null;
  }
  if (path.startsWith("docs/")) {
    return path.slice("docs/".length).includes("/") ? null : path;
  }
  return null;
}

/** True for absolute URLs (`https:`, `data:`, …) and root-relative paths, which are used verbatim. */
function isAbsoluteRef(ref: string): boolean {
  return /^[a-z][a-z0-9+.-]*:/i.test(ref) || ref.startsWith("/");
}

/**
 * Resolve a GitHub-relative path against the directory of the page it appears on
 * (manual chapters resolve against docs/manual/, reference docs against docs/).
 * Returns the repo-relative path, or null when it climbs above the repo root.
 */
function resolveRepoPath(relative: string, currentPage: string): string | null {
  const segments = pageToRepoPath(currentPage).split("/").slice(0, -1);
  for (const seg of relative.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (segments.length === 0) return null;
      segments.pop();
      continue;
    }
    segments.push(seg);
  }
  return segments.join("/");
}

/**
 * Resolve an in-app page for a GitHub-relative href on `currentPage`.
 * Returns the target location for internal navigation, or null for external/default handling.
 */
function resolveManualLink(href: string, currentPage: string): ManualLocation | null {
  const { path: pathPart, hash } = stripHash(href.trim());
  if (!pathPart || isAbsoluteRef(pathPart)) return null;

  const repoPath = resolveRepoPath(pathPart, currentPage);
  const page = repoPath === null ? null : repoPathToPage(repoPath);
  if (!page) return null;
  return hash ? { path: page, hash } : { path: page };
}

/**
 * Served URL for an image referenced from `currentPage`. Only docs/manual/** is mirrored into the
 * app (as /manual/**), so other repo-relative images resolve to null.
 */
function resolveManualImage(src: string, currentPage: string): string | null {
  const s = src.trim();
  if (!s) return null;
  if (isAbsoluteRef(s)) return s;
  const repoPath = resolveRepoPath(s, currentPage);
  if (repoPath === null || !repoPath.startsWith("docs/manual/")) return null;
  return `/manual/${repoPath.slice("docs/manual/".length)}`;
}

const mdComponents: Components = {
  // Keep the `id` rehype-slug adds: `#anchor` links and cross-page hashes scroll to it via getElementById.
  h1: ({ id, children }) => <h1 id={id} className="text-xl font-bold mt-4 mb-2 text-base-content">{children}</h1>,
  h2: ({ id, children }) => <h2 id={id} className="text-lg font-semibold mt-4 mb-2 text-base-content">{children}</h2>,
  h3: ({ id, children }) => <h3 id={id} className="text-base font-semibold mt-3 mb-1 text-base-content">{children}</h3>,
  h4: ({ id, children }) => <h4 id={id} className="text-sm font-semibold mt-3 mb-1 text-base-content">{children}</h4>,
  p: ({ children }) => <p className="text-sm text-base-content/90 mb-2 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-5 text-sm text-base-content/90 mb-2 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 text-sm text-base-content/90 mb-2 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  code: ({ className, children }) => {
    const inline = !className;
    if (inline) {
      return <code className="bg-base-300 px-1 py-0.5 rounded text-xs font-mono">{children}</code>;
    }
    return <code className={className}>{children}</code>;
  },
  pre: ({ children }) => (
    <pre className="bg-base-300 rounded-lg p-3 text-xs font-mono overflow-x-auto mb-3 border border-base-content/10">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto mb-3 border border-base-content/10 rounded-lg">
      <table className="table table-sm w-full">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead>{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr>{children}</tr>,
  th: ({ children }) => <th>{children}</th>,
  td: ({ children }) => <td>{children}</td>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-primary/40 pl-3 my-2 text-sm text-base-content/80">{children}</blockquote>
  ),
};

interface HelpManualModalProps {
  open: boolean;
  onClose: () => void;
}

export const HelpManualModal: FC<HelpManualModalProps> = memo(({ open, onClose }) => {
  const { t } = useTranslation();
  const [loc, setLoc] = useState<ManualLocation>({ path: "README.md" });
  /** Last fetch result, tagged with the page it belongs to so stale bodies never scroll or render. */
  const [page, setPage] = useState<{ path: string; body: string; error: string | null } | null>(null);

  useEffect(() => {
    if (!open) return;
    const path = loc.path;
    let cancelled = false;
    setPage(null);
    void (async () => {
      try {
        const res = await fetch(manualFetchUrl(path), { cache: "no-cache" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const body = await res.text();
        if (!cancelled) setPage({ path, body, error: null });
      } catch (e: unknown) {
        if (!cancelled) setPage({ path, body: "", error: e instanceof Error ? e.message : "" });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, loc.path]);

  useEffect(() => {
    if (!loc.hash || !page || page.error !== null || page.path !== loc.path) return;
    const id = decodeURIComponent(loc.hash);
    const timer = window.setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [page, loc]);

  const onLinkClick = useCallback(
    (e: React.MouseEvent, href: string) => {
      if (href.startsWith("#")) {
        e.preventDefault();
        const id = decodeURIComponent(href.slice(1));
        document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      const target = resolveManualLink(href, loc.path);
      if (!target) return;
      e.preventDefault();
      setLoc(target);
    },
    [loc.path],
  );

  const components: Components = {
    ...mdComponents,
    a: ({ href, children }) => {
      if (!href) return <span>{children}</span>;
      const isHashOnly = href.startsWith("#");
      const internal = isHashOnly || resolveManualLink(href, loc.path) !== null;
      return (
        <a
          href={href}
          className="text-primary underline hover:no-underline"
          onClick={(e) => internal && onLinkClick(e, href)}
          {...(internal ? {} : { target: "_blank", rel: "noreferrer" })}
        >
          {children}
        </a>
      );
    },
    img: ({ src, alt }) => {
      const resolved = resolveManualImage(typeof src === "string" ? src : "", loc.path);
      if (!resolved) return alt ? <span className="text-xs italic text-base-content/50">{alt}</span> : null;
      return <img src={resolved} alt={alt ?? ""} className="rounded-lg border border-base-content/10 max-w-full h-auto my-3" />;
    },
  };

  const isSidebarPage = MANUAL_PAGES.some((p) => p.file === loc.path);
  const loaded = page !== null && page.path === loc.path ? page : null;

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/50 z-[60]" onClick={onClose} aria-hidden />
      <div
        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-200 rounded-xl shadow-2xl z-[61] w-[min(56rem,calc(100vw-2rem))] h-[min(40rem,calc(100vh-2rem))] flex flex-col border border-base-content/10"
        role="dialog"
        aria-label={t("manual.title")}
      >
        <div className="flex-none flex items-center justify-between gap-2 px-4 py-3 border-b border-base-content/10">
          <div className="flex items-center gap-2 min-w-0">
            <RiBookReadLine className="text-lg text-primary flex-shrink-0" aria-hidden />
            <h2 className="text-sm font-bold truncate">{t("manual.title")}</h2>
          </div>
          <button
            type="button"
            className="btn btn-sm btn-circle btn-ghost"
            onClick={onClose}
            title={t("manual.close")}
            aria-label={t("manual.close")}
          >
            <RiCloseLine className="text-lg" />
          </button>
        </div>
        <div className="flex flex-1 min-h-0">
          <nav className="w-48 flex-none border-r border-base-content/10 overflow-y-auto p-2 hidden ui-sm:block">
            {MANUAL_PAGES.map((p) => (
              <button
                key={p.file}
                type="button"
                onClick={() => setLoc({ path: p.file })}
                className={`w-full text-left text-xs px-2 py-1.5 rounded-md truncate ${
                  loc.path === p.file ? "bg-primary/20 text-primary font-medium" : "hover:bg-base-content/5 text-base-content/80"
                }`}
              >
                {t(p.labelKey)}
              </button>
            ))}
          </nav>
          <div className="flex-1 min-w-0 overflow-y-auto p-4">
            {!isSidebarPage && (
              <button type="button" className="btn btn-xs btn-ghost mb-2" onClick={() => setLoc({ path: "README.md" })}>
                {t("manual.back")}
              </button>
            )}
            {loaded && loaded.error !== null && (
              <p className="text-sm text-error mb-2">
                {loaded.error ? `${t("manual.load_failed")}: ${loaded.error}` : t("manual.load_failed")}
              </p>
            )}
            {loaded && loaded.error === null && (
              <article className="manual-md max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSlug]} components={components}>
                  {loaded.body}
                </ReactMarkdown>
              </article>
            )}
            {!loaded && <p className="text-sm text-base-content/50">{t("manual.loading")}</p>}
          </div>
        </div>
        <div className="flex-none px-4 py-2 border-t border-base-content/10 text-[10px] text-base-content/40">
          <Trans
            t={t}
            i18nKey="manual.footer"
            values={{ source: "docs/manual", dest: "public/manual" }}
            components={{ code: <code className="font-mono" /> }}
          />
        </div>
      </div>
    </>
  );
});

HelpManualModal.displayName = "HelpManualModal";
