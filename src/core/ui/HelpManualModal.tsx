import { FC, memo, useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import { RiBookReadLine, RiCloseLine } from "react-icons/ri";

export const MANUAL_PAGES: { label: string; file: string }[] = [
  { label: "Manual home", file: "README.md" },
  { label: "Getting started", file: "getting-started.md" },
  { label: "UI overview", file: "ui-overview.md" },
  { label: "Client mirror", file: "client.md" },
  { label: "Speech to text", file: "stt.md" },
  { label: "Text to speech", file: "tts.md" },
  { label: "Translation", file: "translation.md" },
  { label: "AI transform", file: "transform.md" },
  { label: "Voice changer", file: "voice-changer.md" },
  { label: "Canvas and elements", file: "canvas.md" },
  { label: "Files", file: "files.md" },
  { label: "Twitch", file: "twitch.md" },
  { label: "Kick", file: "kick.md" },
  { label: "Discord", file: "discord.md" },
  { label: "OBS Studio", file: "obs.md" },
  { label: "VRChat", file: "vrc.md" },
  { label: "Project", file: "project.md" },
  { label: "Settings", file: "settings.md" },
  { label: "Shortcuts and tips", file: "shortcuts-and-tips.md" },
];

function manualFetchUrl(name: string): string {
  if (name === "BACKENDS.md") return "/BACKENDS.md";
  if (name === "repo-README.md") return "/repo-README.md";
  if (name === "ARCHITECTURE.md") return "/ARCHITECTURE.md";
  const parts = name.split("/").map((seg) => encodeURIComponent(seg)).join("/");
  return `/manual/${parts}`;
}

/** Resolve in-app fetch key for href relative to docs/manual (GitHub-relative links). */
function resolveManualLink(href: string): string | null {
  const t = href.trim().split("#")[0];
  if (!t || t.startsWith("#")) return null;
  if (/^https?:\/\//i.test(t)) return null;

  let depth = 0;
  let u = t;
  while (u.startsWith("../")) {
    depth++;
    u = u.slice(3);
  }
  u = u.replace(/^\.\//, "");
  if (!u.endsWith(".md")) return null;

  if (u === "BACKENDS.md" && depth >= 1) return "BACKENDS.md";
  if (u === "README.md" && depth >= 2) return "repo-README.md";
  if (u === "ARCHITECTURE.md" && depth >= 2) return "ARCHITECTURE.md";

  if (depth === 0 && !u.includes("..")) return u;
  return null;
}

const mdComponents: Components = {
  h1: ({ children }) => <h1 className="text-xl font-bold mt-4 mb-2 text-base-content">{children}</h1>,
  h2: ({ children }) => <h2 className="text-lg font-semibold mt-4 mb-2 text-base-content">{children}</h2>,
  h3: ({ children }) => <h3 className="text-base font-semibold mt-3 mb-1 text-base-content">{children}</h3>,
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
  img: ({ src, alt }) => {
    const s = typeof src === "string" ? src : "";
    const resolved =
      s.startsWith("http") || s.startsWith("/") ? s : `/manual/${s.replace(/^\.\//, "")}`;
    return <img src={resolved} alt={alt ?? ""} className="rounded-lg border border-base-content/10 max-w-full h-auto my-3" />;
  },
};

interface HelpManualModalProps {
  open: boolean;
  onClose: () => void;
}

export const HelpManualModal: FC<HelpManualModalProps> = memo(({ open, onClose }) => {
  const [file, setFile] = useState("README.md");
  const [body, setBody] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);

  const fetchPage = useCallback(async (name: string) => {
    setErr(null);
    setBody("");
    try {
      const res = await fetch(manualFetchUrl(name), { cache: "no-cache" });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setBody(await res.text());
    } catch (e: unknown) {
      setBody("");
      setErr(e instanceof Error ? e.message : "Failed to load manual");
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void fetchPage(file);
  }, [open, file, fetchPage]);

  const onLinkClick = useCallback((e: React.MouseEvent, href: string) => {
    const name = resolveManualLink(href);
    if (!name) return;
    e.preventDefault();
    setFile(name);
  }, []);

  const components: Components = {
    ...mdComponents,
    a: ({ href, children }) => {
      if (!href) return <span>{children}</span>;
      const internal = resolveManualLink(href);
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
  };

  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/50 z-[60]" onClick={onClose} aria-hidden />
      <div
        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-base-200 rounded-xl shadow-2xl z-[61] w-[min(56rem,calc(100vw-2rem))] h-[min(40rem,calc(100vh-2rem))] flex flex-col border border-base-content/10"
        role="dialog"
        aria-label="User manual"
      >
        <div className="flex-none flex items-center justify-between gap-2 px-4 py-3 border-b border-base-content/10">
          <div className="flex items-center gap-2 min-w-0">
            <RiBookReadLine className="text-lg text-primary flex-shrink-0" aria-hidden />
            <h2 className="text-sm font-bold truncate">User manual</h2>
          </div>
          <button type="button" className="btn btn-sm btn-circle btn-ghost" onClick={onClose} title="Close" aria-label="Close">
            <RiCloseLine className="text-lg" />
          </button>
        </div>
        <div className="flex flex-1 min-h-0">
          <nav className="w-48 flex-none border-r border-base-content/10 overflow-y-auto p-2 hidden sm:block">
            {MANUAL_PAGES.map((p) => (
              <button
                key={p.file}
                type="button"
                onClick={() => setFile(p.file)}
                className={`w-full text-left text-xs px-2 py-1.5 rounded-md truncate ${
                  p.file === file ? "bg-primary/20 text-primary font-medium" : "hover:bg-base-content/5 text-base-content/80"
                }`}
              >
                {p.label}
              </button>
            ))}
          </nav>
          <div className="flex-1 min-w-0 overflow-y-auto p-4">
            {file === "BACKENDS.md" && (
              <button type="button" className="btn btn-xs btn-ghost mb-2" onClick={() => setFile("README.md")}>
                Back to manual
              </button>
            )}
            {err && <p className="text-sm text-error mb-2">{err}</p>}
            {!err && body && (
              <article className="manual-md max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                  {body}
                </ReactMarkdown>
              </article>
            )}
            {!err && !body && <p className="text-sm text-base-content/50">Loading…</p>}
          </div>
        </div>
        <div className="flex-none px-4 py-2 border-t border-base-content/10 text-[10px] text-base-content/40">
          Source: <code className="font-mono">docs/manual</code> (synced to <code className="font-mono">public/manual</code> on dev/build).
        </div>
      </div>
    </>
  );
});

HelpManualModal.displayName = "HelpManualModal";
