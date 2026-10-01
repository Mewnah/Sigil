//! The flow-matching sampler of Sigil-ZS: upstream `BASECFM.solve_euler` with classifier-free
//! guidance, driven over the exported `dit_step` graph (ADR 0006; `sampler.py` `SigilZS.cfm`).
//!
//! The step graph does one Euler step with CFG inside (the conditional and unconditional passes as
//! one batch of 2, `x + dt · ((1 + cfg) v_c − cfg v_u)`, then the prompt frames zeroed again);
//! `dit_step1` is the `cfg = 0` step (batch 1). The host only walks the time grid: `t` starts at
//! `t_span[0] = 0`, every step takes `dt = t_span[k] − t_span[k − 1]` from the f32 `linspace` and
//! accumulates `t += dt` in f32, exactly as upstream (so the graph sees bit-identical `t` / `dt`), and
//! feeds `x_next` back as the next `x`.
//!
//! Graphs are driven through [`Stage`] (one fixed-shape session: input buffers, a run, output
//! buffers), implemented by `gpu::SessionPlan` and by the tests' analytic flows.

use crate::services::neural_audio::gpu::SessionPlan;

use super::dsp::linspace_f32;

/// One fixed-shape graph: write the inputs, run, read the outputs (`gpu::SessionPlan`'s API).
pub trait Stage: Send {
    fn input_f32(&mut self, name: &str) -> Result<&mut [f32], String>;
    fn input_i64(&mut self, name: &str) -> Result<&mut [i64], String>;
    fn run(&mut self) -> Result<(), String>;
    fn output(&self, name: &str) -> Result<&[f32], String>;
}

impl Stage for SessionPlan {
    fn input_f32(&mut self, name: &str) -> Result<&mut [f32], String> {
        SessionPlan::input_f32(self, name)
    }

    fn input_i64(&mut self, name: &str) -> Result<&mut [i64], String> {
        SessionPlan::input_i64(self, name)
    }

    fn run(&mut self) -> Result<(), String> {
        SessionPlan::run(self)
    }

    fn output(&self, name: &str) -> Result<&[f32], String> {
        SessionPlan::output(self, name)
    }
}

/// Run `steps` Euler steps of the flow on `step` (a `dit_step` / `dit_step1` stage whose `x`,
/// `prompt_x`, `cond`, `style`, `prompt_len` and, for CFG, `cfg_rate` inputs are written). `x` is the
/// starting noise (prompt frames zero) and receives the result (`[1, 80, L]`, channel-major).
/// Allocation-free once `t_span` is computed (`linspace_f32(steps)`).
pub fn solve_euler(step: &mut dyn Stage, t_span: &[f32], x: &mut [f32]) -> Result<(), String> {
    if t_span.len() < 2 {
        return Err("the flow needs at least one step".into());
    }
    let mut t = t_span[0];
    for k in 1..t_span.len() {
        let dt = t_span[k] - t_span[k - 1];
        {
            let xin = step.input_f32("x")?;
            if xin.len() != x.len() {
                return Err(format!("dit_step: x holds {} values, the state {}", xin.len(), x.len()));
            }
            xin.copy_from_slice(x);
        }
        step.input_f32("t")?[0] = t;
        step.input_f32("dt")?[0] = dt;
        step.run()?;
        let next = step.output("x_next")?;
        if next.len() != x.len() {
            return Err(format!("dit_step: x_next holds {} values, the state {}", next.len(), x.len()));
        }
        x.copy_from_slice(next);
        t += dt;
    }
    Ok(())
}

/// The time grid of `steps` Euler steps (`linspace_f32`).
pub fn time_grid(steps: u32) -> Vec<f32> {
    linspace_f32(steps.max(1))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An analytic `dit_step`: the rectified flow toward a target `a` (conditional) and `b`
    /// (unconditional), `v = ((1 + cfg)(a − x) − cfg (b − x)) / (1 − t)`, with the prompt frames
    /// re-zeroed. Euler integration of this flow from any `x` lands exactly on `(1 + cfg) a − cfg b`
    /// at `t = 1` (each step moves `x` the same fraction `dt / (1 − t)` of the way). It records every
    /// `(t, dt)` it was given.
    pub(crate) struct AnalyticStep {
        pub mel: usize,
        pub l: usize,
        pub prompt: usize,
        pub cfg: f32,
        pub a: f32,
        pub b: f32,
        pub x: Vec<f32>,
        pub t: [f32; 1],
        pub dt: [f32; 1],
        pub next: Vec<f32>,
        pub calls: Vec<(f32, f32)>,
    }

    impl AnalyticStep {
        pub(crate) fn new(mel: usize, l: usize, prompt: usize, cfg: f32, a: f32, b: f32) -> AnalyticStep {
            AnalyticStep { mel, l, prompt, cfg, a, b, x: vec![0.0; mel * l], t: [0.0], dt: [0.0], next: vec![0.0; mel * l], calls: Vec::new() }
        }
    }

    impl Stage for AnalyticStep {
        fn input_f32(&mut self, name: &str) -> Result<&mut [f32], String> {
            match name {
                "x" => Ok(&mut self.x),
                "t" => Ok(&mut self.t),
                "dt" => Ok(&mut self.dt),
                _ => Err(format!("no input '{name}'")),
            }
        }

        fn input_i64(&mut self, name: &str) -> Result<&mut [i64], String> {
            Err(format!("no input '{name}'"))
        }

        fn run(&mut self) -> Result<(), String> {
            let (t, dt) = (self.t[0], self.dt[0]);
            self.calls.push((t, dt));
            for c in 0..self.mel {
                for j in 0..self.l {
                    let i = c * self.l + j;
                    let x = self.x[i];
                    let v = ((1.0 + self.cfg) * (self.a - x) - self.cfg * (self.b - x)) / (1.0 - t);
                    self.next[i] = if j < self.prompt { 0.0 } else { x + dt * v };
                }
            }
            Ok(())
        }

        fn output(&self, name: &str) -> Result<&[f32], String> {
            match name {
                "x_next" => Ok(&self.next),
                _ => Err(format!("no output '{name}'")),
            }
        }
    }

    #[test]
    fn euler_on_an_analytic_flow_lands_on_its_target() {
        for (steps, cfg) in [(1u32, 0.0f32), (4, 0.0), (6, 0.7), (10, 0.7), (30, 1.0)] {
            let (mel, l, prompt) = (3, 11, 4);
            let mut step = AnalyticStep::new(mel, l, prompt, cfg, 0.25, -0.5);
            let mut x: Vec<f32> = (0..mel * l).map(|i| if i % l < prompt { 0.0 } else { (i as f32 * 0.37).sin() }).collect();
            let grid = time_grid(steps);
            solve_euler(&mut step, &grid, &mut x).unwrap();
            let target = (1.0 + cfg) * 0.25 - cfg * -0.5;
            for c in 0..mel {
                for j in 0..l {
                    let v = x[c * l + j];
                    if j < prompt {
                        assert_eq!(v, 0.0, "prompt frames stay zero");
                    } else {
                        assert!((v - target).abs() < 1e-5, "steps {steps} cfg {cfg}: {v} vs {target}");
                    }
                }
            }
            // One run per step, t accumulated in f32 from the linspace's differences (upstream's
            // arithmetic), dt the linspace's own differences.
            assert_eq!(step.calls.len(), steps as usize);
            let mut t = 0.0f32;
            for (k, &(ct, cdt)) in step.calls.iter().enumerate() {
                assert_eq!(ct, t);
                assert_eq!(cdt, grid[k + 1] - grid[k]);
                t += cdt;
            }
        }
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let mut step = AnalyticStep::new(2, 4, 1, 0.0, 1.0, 0.0);
        let mut x = vec![0.0; 7];
        assert!(solve_euler(&mut step, &time_grid(2), &mut x).is_err());
        let mut x = vec![0.0; 8];
        assert!(solve_euler(&mut step, &[0.0], &mut x).is_err());
    }
}
