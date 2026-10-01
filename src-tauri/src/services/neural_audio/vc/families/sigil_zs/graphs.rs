//! Where the Sigil-ZS graphs come from: the model store (`model-catalog/vc-zeroshot.json`, Sigil's
//! export of Seed-VC `seed-uvit-tat-xlsr-tiny` by `scripts/vc-families/seed_vc/export.py`; entries
//! `unpublished`, installed by the Models tab's Import or `SIGIL_LOCAL_ASSETS`).
//!
//! | entry | files | when |
//! |---|---|---|
//! | [`ZS_GRAPHS`] | every fp32 graph, plus the fp16 decoder (`hift_nsf.fp16.onnx`, `hift.fp16.onnx`) | always |
//! | [`ZS_GRAPHS_FP16`] | `xlsr.fp16.onnx`, `dit_step.fp16.onnx`, `dit_step1.fp16.onnx` | the XLS-R / DiT fp16 switches only |
//!
//! File names are the export's (`<graph>.onnx`, `<graph>.fp16.onnx`), so a folder of exported graphs
//! is also a valid local-assets folder.

use std::path::PathBuf;

/// The fp32 graphs (and the fp16 decoder).
pub const ZS_GRAPHS: &str = "vc/zs-seed-vc";
/// The optional fp16 XLS-R and DiT graphs (switches that fail waveform parity; ADR 0006).
pub const ZS_GRAPHS_FP16: &str = "vc/zs-seed-vc-fp16";

/// One exported graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphName {
    Xlsr,
    Lr,
    DitStep,
    DitStep1,
    Hift,
    HiftF0,
    HiftNsf,
    Campplus,
    Mel,
}

impl GraphName {
    pub fn stem(self) -> &'static str {
        match self {
            GraphName::Xlsr => "xlsr",
            GraphName::Lr => "lr",
            GraphName::DitStep => "dit_step",
            GraphName::DitStep1 => "dit_step1",
            GraphName::Hift => "hift",
            GraphName::HiftF0 => "hift_f0",
            GraphName::HiftNsf => "hift_nsf",
            GraphName::Campplus => "campplus",
            GraphName::Mel => "mel",
        }
    }

    /// Whether the export has an fp16 variant (fp16 inside, fp32 I/O).
    pub fn has_fp16(self) -> bool {
        matches!(self, GraphName::Xlsr | GraphName::DitStep | GraphName::DitStep1 | GraphName::Hift | GraphName::HiftNsf)
    }

    /// File name of the variant.
    pub fn file(self, fp16: bool) -> String {
        if fp16 && self.has_fp16() {
            format!("{}.fp16.onnx", self.stem())
        } else {
            format!("{}.onnx", self.stem())
        }
    }

    /// Which entry holds the variant.
    fn entry(self, fp16: bool) -> &'static str {
        if fp16 && matches!(self, GraphName::Xlsr | GraphName::DitStep | GraphName::DitStep1) {
            ZS_GRAPHS_FP16
        } else {
            ZS_GRAPHS
        }
    }
}

/// Files every voice needs from [`ZS_GRAPHS`] (the vocoder in either mode, the step graphs for
/// either CFG setting, the reference graphs).
pub const REQUIRED: [GraphName; 9] = [
    GraphName::Xlsr,
    GraphName::Lr,
    GraphName::DitStep,
    GraphName::DitStep1,
    GraphName::Hift,
    GraphName::HiftF0,
    GraphName::HiftNsf,
    GraphName::Campplus,
    GraphName::Mel,
];

/// The folders the graphs are read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphSet {
    pub main: PathBuf,
    /// The fp16 entry's folder, when installed.
    pub fp16: Option<PathBuf>,
}

impl GraphSet {
    /// The installed store entries; an error names the missing entry.
    pub fn installed() -> Result<GraphSet, String> {
        let main = crate::services::model_store::installed_path(ZS_GRAPHS).ok_or_else(|| {
            format!("the Sigil-ZS graphs ({ZS_GRAPHS}) are not installed; import them in Models & Runtimes")
        })?;
        let set = GraphSet { main, fp16: crate::services::model_store::installed_path(ZS_GRAPHS_FP16) };
        set.check()?;
        Ok(set)
    }

    /// Every graph in one folder (exported graphs; the tests).
    #[cfg(test)]
    pub fn from_dir(dir: &std::path::Path) -> GraphSet {
        GraphSet { main: dir.to_path_buf(), fp16: Some(dir.to_path_buf()) }
    }

    /// Every required fp32 graph is present.
    pub fn check(&self) -> Result<(), String> {
        for g in REQUIRED {
            let p = self.main.join(g.file(false));
            if !p.is_file() {
                return Err(format!("the Sigil-ZS graph {} is missing ({}); reinstall {ZS_GRAPHS}", g.file(false), p.display()));
            }
        }
        Ok(())
    }

    /// The file of `graph` (`fp16` when available, else fp32 with the reason).
    pub fn path(&self, graph: GraphName, fp16: bool) -> (PathBuf, Option<String>) {
        if fp16 && graph.has_fp16() {
            let dir = match graph.entry(true) {
                ZS_GRAPHS => Some(&self.main),
                _ => self.fp16.as_ref(),
            };
            match dir.map(|d| d.join(graph.file(true))).filter(|p| p.is_file()) {
                Some(p) => return (p, None),
                None => {
                    return (
                        self.main.join(graph.file(false)),
                        Some(format!("{}: the fp16 graph is not installed ({}); running fp32", graph.stem(), graph.entry(true))),
                    )
                }
            }
        }
        (self.main.join(graph.file(false)), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_and_fallbacks() {
        assert_eq!(GraphName::Xlsr.file(true), "xlsr.fp16.onnx");
        assert_eq!(GraphName::HiftF0.file(true), "hift_f0.onnx", "hift_f0 is fp32 only");
        assert_eq!(GraphName::Mel.file(false), "mel.onnx");
        let dir = std::env::temp_dir().join(format!("sigil-zs-graphs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let set = GraphSet { main: dir.clone(), fp16: None };
        assert!(set.check().unwrap_err().contains("xlsr.onnx"));
        for g in REQUIRED {
            std::fs::write(dir.join(g.file(false)), b"x").unwrap();
        }
        set.check().unwrap();
        let (p, why) = set.path(GraphName::DitStep, true);
        assert_eq!(p, dir.join("dit_step.onnx"));
        assert!(why.unwrap().contains(ZS_GRAPHS_FP16));
        std::fs::write(dir.join("hift_nsf.fp16.onnx"), b"x").unwrap();
        assert_eq!(set.path(GraphName::HiftNsf, true), (dir.join("hift_nsf.fp16.onnx"), None), "the fp16 decoder ships in the main entry");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
