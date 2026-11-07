mod generator;
mod learn;
mod model;
mod profiles;
mod session;
mod sim;
mod storage;
mod typing;
mod ui;
mod voice;

use anyhow::{Context, Result, bail};

const USAGE: &str = "usage: thehill [--simulate N [--slow th,er]] [--retrain] [--stats] [--calibrate [N]]";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args.get(i + 1));
    storage::adopt_legacy_data().context("could not copy data over from Sisyphus")?;
    let dir = profiles::dir_of(&profiles::active());

    if let Some(n) = flag("--simulate") {
        let n: usize = n.context(USAGE)?.parse().context("--simulate takes a run count")?;
        let slow: Vec<String> = match flag("--slow") {
            Some(list) => list.context(USAGE)?.split(',').map(String::from).collect(),
            None => vec!["th".into(), "er".into()],
        };
        // Never mix synthetic runs into real data unless asked to (then into the active profile).
        let dir = match std::env::var_os("THEHILL_DATA_DIR") {
            Some(_) => dir,
            None => std::env::temp_dir().join(format!("thehill-sim-{}", std::process::id())),
        };
        println!("simulating {n} runs in {}", dir.display());
        sim::simulate(&dir, n, &slow, false)?;
        return Ok(());
    }
    if args.iter().any(|a| a == "--retrain") {
        learn::retrain(&storage::Store::open(&dir)?)?;
        return Ok(());
    }
    if let Some(n) = flag("--calibrate") {
        let n = n.map_or(Ok(20), |n| n.parse()).context("--calibrate takes a run count")?;
        return calibrate(&dir, n);
    }
    if args.iter().any(|a| a == "--stats") {
        return stats(&dir);
    }
    if !args.is_empty() {
        bail!(USAGE);
    }
    ui::run(session::Session::open(&dir)?);
    Ok(())
}

/// Compares the live model's predictions with what actually happened in the last `n` runs.
fn calibrate(dir: &std::path::Path, n: usize) -> Result<()> {
    let store = storage::Store::open(dir)?;
    learn::migrate(&store)?;
    let model = learn::load_live(&store).context("no live model")?;
    println!("error_ms {:.0}  ms_scale {:.3}", model.error_ms, model.ms_scale);
    println!(
        "{:>8} {:>8} {:>8} {:>7} {:>7} {:>6} {:>6}  text",
        "pred_ms", "act_ms", "logbias", "p_err", "err", "pred", "wpm"
    );
    for run in store.recent_runs(n)? {
        let c = model.calibrate(&run);
        let text: String = run.text.iter().take(50).collect();
        let pred_wpm = model.predicted_wpm(run.reached());
        println!(
            "{:>8.0} {:>8.0} {:>8.2} {:>7.3} {:>7.3} {:>6.0} {:>6.0} {:>4}  {text}",
            c.predicted_ms,
            c.actual_ms,
            c.log_bias,
            c.predicted_err,
            c.actual_err,
            pred_wpm,
            run.wpm,
            if run.hill { "hill" } else { "" }
        );
    }
    Ok(())
}

fn stats(dir: &std::path::Path) -> Result<()> {
    let store = storage::Store::open(dir)?;
    println!("data: {}", dir.display());
    println!("runs: {}", store.run_count()?);
    for c in store.checkpoints()? {
        println!("checkpoint {}: {}  (reference {:.0} wpm)", c.id, c.label(), c.reference_wpm);
    }
    if let Some(model) = learn::load_live(&store) {
        println!("context: {} chars", model.context());
        println!("live model reference: {:.0} wpm", learn::reference_wpm(&model));
        println!("weakest transitions: {}", learn::weaknesses(&model, 10).join("  "));
        let costs = generator::WordCosts::new(&model);
        println!("hardest words: {}", costs.hardest(15).collect::<Vec<_>>().join(" "));
    }
    Ok(())
}
