//! The VRAM claims ledger: who holds (or is about to load) how much video memory.
//!
//! [`Ledger`] is plain data (unit-tested directly); the process-wide instance lives behind a mutex
//! in `resources/mod.rs`, which wraps it in RAII [`super::VramClaim`] handles. Claims are estimates
//! a subsystem registers before loading a model and resizes once it knows the real footprint;
//! `external` claims belong to helper processes (llama-server, TTS servers), whose memory never
//! shows up in this process's DXGI usage.

use std::collections::BTreeMap;

use super::{Subsystem, VramClaimInfo};

/// Longest label kept (labels are for people: "rvc: Mika v2", "llama-server: hy-mt2-7b").
pub const MAX_LABEL_CHARS: usize = 80;

#[derive(Debug, Default)]
pub struct Ledger {
    next_id: u64,
    claims: BTreeMap<u64, VramClaimInfo>,
}

impl Ledger {
    pub const fn new() -> Ledger {
        Ledger {
            next_id: 0,
            claims: BTreeMap::new(),
        }
    }

    /// Register a claim; ids start at 1 and are never reused within a process.
    pub fn insert(&mut self, subsystem: Subsystem, label: &str, mb: u64, external: bool) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        let label: String = label.trim().chars().take(MAX_LABEL_CHARS).collect();
        self.claims.insert(
            id,
            VramClaimInfo {
                id,
                subsystem,
                label,
                mb,
                external,
            },
        );
        id
    }

    /// Change a claim's size; `true` when something changed.
    pub fn resize(&mut self, id: u64, mb: u64) -> bool {
        match self.claims.get_mut(&id) {
            Some(c) if c.mb != mb => {
                c.mb = mb;
                true
            }
            _ => false,
        }
    }

    /// Release a claim; `true` when it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        self.claims.remove(&id).is_some()
    }

    /// Every live claim, oldest first.
    pub fn list(&self) -> Vec<VramClaimInfo> {
        self.claims.values().cloned().collect()
    }

    /// `(in-process MiB, helper-process MiB)`.
    pub fn totals(&self) -> (u64, u64) {
        self.claims.values().fold((0, 0), |(inner, ext), c| {
            if c.external {
                (inner, ext.saturating_add(c.mb))
            } else {
                (inner.saturating_add(c.mb), ext)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_insert_resize_remove() {
        let mut l = Ledger::new();
        let vc = l.insert(Subsystem::Vc, "  rvc: test voice  ", 1200, false);
        let llm = l.insert(Subsystem::Llm, "llama-server: hy-mt2-7b", 5300, true);
        assert_eq!((vc, llm), (1, 2));
        assert_eq!(l.totals(), (1200, 5300));
        assert_eq!(l.list()[0].label, "rvc: test voice");
        assert!(l.list()[1].external);

        assert!(l.resize(vc, 700));
        assert!(!l.resize(vc, 700), "same size is not a change");
        assert!(!l.resize(99, 1), "unknown id");
        assert_eq!(l.totals(), (700, 5300));

        assert!(l.remove(llm));
        assert!(!l.remove(llm));
        assert_eq!(l.totals(), (700, 0));
        let again = l.insert(Subsystem::Tts, "qwen3-tts", 1500, true);
        assert_eq!(again, 3, "ids are never reused");
        assert_eq!(l.list().iter().map(|c| c.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn labels_are_bounded_and_totals_saturate() {
        let mut l = Ledger::new();
        let id = l.insert(Subsystem::Other, &"x".repeat(500), u64::MAX, false);
        assert_eq!(l.list()[0].label.chars().count(), MAX_LABEL_CHARS);
        l.insert(Subsystem::Other, "more", 10, false);
        assert_eq!(l.totals().0, u64::MAX);
        assert!(l.remove(id));
    }
}
