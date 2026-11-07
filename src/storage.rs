//! SQLite storage. The keystroke log is the source of truth; model files are derived from it.

use crate::model::RunLog;
use crate::typing::{Event, Key};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};

pub fn data_dir() -> PathBuf {
    std::env::var_os("THEHILL_DATA_DIR").map(PathBuf::from).unwrap_or_else(default_dir)
}

fn default_dir() -> PathBuf {
    dirs::data_dir().expect("no data directory on this platform").join("thehill")
}

pub struct Store {
    conn: Connection,
    dir: PathBuf,
}

pub struct RunRecord<'a> {
    pub mode: &'a str,
    pub amount: u32,
    pub punctuation: bool,
    pub numbers: bool,
    pub opponent: Option<i64>,
    pub text: &'a str,
    pub wpm: f64,
    pub raw: f64,
    pub accuracy: f64,
    pub predicted: f64,
    /// Played against the hill (targeted text) rather than random words.
    pub hill: bool,
    /// A boss run: 25 words built from the hill's four worst trigrams.
    pub boss: bool,
    /// The model's own prediction before correcting for how it picks text; 0 if none.
    pub model_predicted: f64,
}

/// An earlier run a word appeared in.
pub struct PastRun {
    pub id: i64,
    pub started_at: i64,
    pub log: RunLog,
}

impl PastRun {
    pub fn date(&self) -> String {
        chrono::DateTime::from_timestamp(self.started_at, 0)
            .map(|d| d.with_timezone(&chrono::Local).format("%b %-d").to_string())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Lifetime {
    pub runs: usize,
    pub seconds: f64,
    pub keystrokes: i64,
    /// Best wpm and the mode it was set in.
    pub best: Option<(f64, String)>,
    pub last10: Option<f64>,
    pub previous10: Option<f64>,
    pub accuracy10: Option<f64>,
    /// Runs against the hill over and under its prediction.
    pub wins: usize,
    pub losses: usize,
    pub longest_streak: usize,
    pub boss_wins: usize,
    pub boss_losses: usize,
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub id: i64,
    pub created_at: i64,
    pub run_count: i64,
    pub tags: Vec<String>,
    pub reference_wpm: f64,
    /// Best result against this checkpoint: (wpm, what it predicted).
    pub best: Option<(f64, f64)>,
    /// Runs played against it.
    pub played: i64,
}

impl Checkpoint {
    pub fn date(&self) -> String {
        chrono::DateTime::from_timestamp(self.created_at, 0)
            .map(|d| d.with_timezone(&chrono::Local).format("%b %-d").to_string())
            .unwrap_or_default()
    }

    pub fn label(&self) -> String {
        format!("{} · run {} · {}", self.date(), self.run_count, self.tags.join(" "))
    }
}

fn key_code(key: Key) -> String {
    match key {
        Key::Char(c) => c.to_string(),
        Key::Backspace => "\u{8}".into(),
        Key::ClearWord => "\u{7f}".into(),
    }
}

fn key_from(code: &str) -> Key {
    match code {
        "\u{8}" => Key::Backspace,
        "\u{7f}" => Key::ClearWord,
        s => Key::Char(s.chars().next().unwrap_or(' ')),
    }
}

impl Store {
    /// Opens an existing profile's database without creating anything: for work that must
    /// not bring back a profile that was deleted meanwhile.
    pub fn open_existing(dir: &Path) -> Result<Store> {
        if !dir.join("thehill.db").is_file() {
            anyhow::bail!("{} has no database", dir.display());
        }
        Self::open(dir)
    }

    pub fn open(dir: &Path) -> Result<Store> {
        std::fs::create_dir_all(dir.join("checkpoints"))?;
        let conn = Connection::open(dir.join("thehill.db"))?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS runs (
                 id INTEGER PRIMARY KEY,
                 started_at INTEGER NOT NULL,
                 mode TEXT NOT NULL,
                 amount INTEGER NOT NULL,
                 punctuation INTEGER NOT NULL,
                 numbers INTEGER NOT NULL,
                 opponent INTEGER,
                 text TEXT NOT NULL,
                 wpm REAL NOT NULL,
                 raw REAL NOT NULL,
                 accuracy REAL NOT NULL,
                 predicted REAL NOT NULL,
                 hill INTEGER NOT NULL,
                 boss INTEGER NOT NULL,
                 model_predicted REAL NOT NULL
             );
             CREATE TABLE IF NOT EXISTS keys (
                 run_id INTEGER NOT NULL REFERENCES runs(id),
                 seq INTEGER NOT NULL,
                 t REAL NOT NULL,
                 idx INTEGER NOT NULL,
                 expected TEXT,
                 key TEXT NOT NULL,
                 correct INTEGER NOT NULL,
                 PRIMARY KEY (run_id, seq)
             );
             CREATE TABLE IF NOT EXISTS checkpoints (
                 id INTEGER PRIMARY KEY,
                 created_at INTEGER NOT NULL,
                 run_count INTEGER NOT NULL,
                 tags TEXT NOT NULL,
                 reference_wpm REAL NOT NULL
             );
             CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )?;
        Ok(Store { conn, dir: dir.to_path_buf() })
    }

    pub fn live_model_path(&self) -> PathBuf {
        self.dir.join("live.bin")
    }

    pub fn checkpoint_path(&self, id: i64) -> PathBuf {
        self.dir.join("checkpoints").join(format!("{id}.bin"))
    }

    pub fn save_run(&mut self, run: &RunRecord, events: &[Event]) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO runs (started_at, mode, amount, punctuation, numbers, opponent, text, wpm, raw, accuracy, predicted, hill, boss, model_predicted)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                chrono::Utc::now().timestamp(),
                run.mode,
                run.amount,
                run.punctuation,
                run.numbers,
                run.opponent,
                run.text,
                run.wpm,
                run.raw,
                run.accuracy,
                run.predicted,
                run.hill,
                run.boss,
                run.model_predicted
            ],
        )?;
        let id = tx.last_insert_rowid();
        {
            let mut insert = tx.prepare(
                "INSERT INTO keys (run_id, seq, t, idx, expected, key, correct) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for (seq, e) in events.iter().enumerate() {
                insert.execute(params![
                    id,
                    seq,
                    e.t,
                    e.idx,
                    e.expected.map(String::from),
                    key_code(e.key),
                    e.correct
                ])?;
            }
        }
        tx.commit()?;
        Ok(id)
    }

    pub fn run_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM runs", [], |r| r.get(0))?)
    }

    /// The last `limit` runs and the total run count, read from one consistent snapshot so a
    /// run logged meanwhile can't be counted without being trained on.
    pub fn training_snapshot(&self, limit: usize) -> Result<(Vec<RunLog>, i64)> {
        let tx = self.conn.unchecked_transaction()?;
        let runs = self.recent_runs(limit)?;
        let count = self.run_count()?;
        tx.commit()?;
        Ok((runs, count))
    }

    /// The last `limit` runs, oldest first.
    pub fn recent_runs(&self, limit: usize) -> Result<Vec<RunLog>> {
        self.load_runs("SELECT id, text, wpm, hill FROM runs ORDER BY id DESC LIMIT ?1", limit as i64)
    }

    /// Every run's mode, amount, and whether it was a boss or against a checkpoint; oldest
    /// first, like `recent_runs`.
    pub fn run_kinds(&self) -> Result<Vec<(String, i64, bool, bool)>> {
        let mut stmt = self.conn.prepare("SELECT mode, amount, boss, opponent IS NOT NULL FROM runs ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Runs for a query yielding (id, text, wpm, hill), returned in reverse query order.
    fn load_runs(&self, sql: &str, param: i64) -> Result<Vec<RunLog>> {
        let mut runs = self.conn.prepare(sql)?;
        let mut keys =
            self.conn.prepare("SELECT t, idx, expected, key, correct FROM keys WHERE run_id = ?1 ORDER BY seq")?;
        let mut out = Vec::new();
        let rows: Vec<(i64, String, f64, bool)> =
            runs.query_map([param], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?.collect::<Result<_, _>>()?;
        for (id, text, wpm, hill) in rows.into_iter().rev() {
            let events = keys
                .query_map([id], |r| {
                    Ok(Event {
                        t: r.get(0)?,
                        idx: r.get::<_, i64>(1)? as usize,
                        expected: r.get::<_, Option<String>>(2)?.and_then(|s| s.chars().next()),
                        key: key_from(&r.get::<_, String>(3)?),
                        correct: r.get(4)?,
                    })
                })?
                .collect::<Result<_, _>>()?;
            out.push(RunLog { text: text.chars().collect(), events, wpm, hill });
        }
        Ok(out)
    }

    /// Records a checkpoint and writes its model with `save`. The row only commits if the
    /// write succeeds, so there are never rows without models.
    pub fn add_checkpoint(
        &self,
        run_count: i64,
        tags: &[String],
        reference_wpm: f64,
        save: impl FnOnce(&Path) -> std::io::Result<()>,
    ) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO checkpoints (created_at, run_count, tags, reference_wpm) VALUES (?1, ?2, ?3, ?4)",
            params![chrono::Utc::now().timestamp(), run_count, tags.join(" "), reference_wpm],
        )?;
        let id = tx.last_insert_rowid();
        save(&self.checkpoint_path(id))?;
        tx.commit()?;
        Ok(id)
    }

    pub fn update_checkpoint(&self, id: i64, tags: &[String], reference_wpm: f64) -> Result<()> {
        self.conn.execute(
            "UPDATE checkpoints SET tags = ?1, reference_wpm = ?2 WHERE id = ?3",
            params![tags.join(" "), reference_wpm, id],
        )?;
        Ok(())
    }

    pub fn checkpoints(&self) -> Result<Vec<Checkpoint>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.created_at, c.run_count, c.tags, c.reference_wpm,
                    (SELECT wpm FROM runs WHERE opponent = c.id AND hill = 1 AND predicted > 0 ORDER BY wpm DESC LIMIT 1),
                    (SELECT predicted FROM runs WHERE opponent = c.id AND hill = 1 AND predicted > 0 ORDER BY wpm DESC LIMIT 1),
                    (SELECT COUNT(*) FROM runs WHERE opponent = c.id AND hill = 1 AND predicted > 0)
             FROM checkpoints c ORDER BY c.id",
        )?;
        let rows = stmt.query_map([], |r| {
            let best: Option<f64> = r.get(5)?;
            let predicted: Option<f64> = r.get(6)?;
            Ok(Checkpoint {
                id: r.get(0)?,
                created_at: r.get(1)?,
                run_count: r.get(2)?,
                tags: r.get::<_, String>(3)?.split(' ').map(String::from).collect(),
                reference_wpm: r.get(4)?,
                best: best.zip(predicted),
                played: r.get(7)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn last_wpm(&self) -> Result<Option<f64>> {
        Ok(self.conn.query_row("SELECT wpm FROM runs ORDER BY id DESC LIMIT 1", [], |r| r.get(0)).optional()?)
    }

    /// Runs finished since local midnight (`started_at` is stamped when a run is saved).
    pub fn runs_today(&self) -> Result<i64> {
        let midnight = chrono::Local::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .and_then(|t| t.and_local_timezone(chrono::Local).earliest())
            .map_or(0, |t| t.timestamp());
        Ok(self.conn.query_row("SELECT COUNT(*) FROM runs WHERE started_at >= ?1", [midnight], |r| r.get(0))?)
    }

    /// Best wpm so far in this exact configuration.
    pub fn best_wpm(&self, mode: &str, amount: u32, punctuation: bool, numbers: bool) -> Result<Option<f64>> {
        Ok(self.conn.query_row(
            "SELECT MAX(wpm) FROM runs WHERE mode = ?1 AND amount = ?2 AND punctuation = ?3 AND numbers = ?4 AND boss = 0",
            params![mode, amount, punctuation, numbers],
            |r| r.get(0),
        )?)
    }

    /// (wpm, predicted) of the last `n` runs against the hill with a prediction, newest first.
    pub fn recent_results(&self, n: usize) -> Result<Vec<(f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT wpm, predicted FROM runs WHERE predicted > 0 AND hill = 1 AND boss = 0 ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([n as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// (wpm, the model's own prediction) of the last `n` ordinary runs against the live hill,
    /// newest first: what the prediction correction learns from.
    pub fn model_results(&self, n: usize) -> Result<Vec<(f64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT wpm, model_predicted FROM runs
             WHERE model_predicted > 0 AND wpm > 0 AND hill = 1 AND boss = 0 AND opponent IS NULL
             ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([n as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn lifetime(&self) -> Result<Lifetime> {
        let mut stmt =
            self.conn.prepare("SELECT wpm, accuracy, predicted, hill, mode, amount, boss FROM runs ORDER BY id")?;
        let runs: Vec<(f64, f64, f64, bool, String, u32, bool)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)))?
            .collect::<Result<_, _>>()?;
        let (seconds, keystrokes): (Option<f64>, i64) = self.conn.query_row(
            "SELECT (SELECT SUM(m) FROM (SELECT MAX(t) AS m FROM keys GROUP BY run_id)) / 1000.0,
                    (SELECT COUNT(*) FROM keys WHERE key NOT IN (char(8), char(127)))",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let mean = |xs: &[f64]| (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64);
        let wpms: Vec<f64> = runs.iter().map(|r| r.0).collect();
        let accs: Vec<f64> = runs.iter().rev().take(10).map(|r| r.1).collect();
        let n = wpms.len();
        let mut life = Lifetime {
            runs: n,
            seconds: seconds.unwrap_or(0.0),
            keystrokes,
            best: runs
                .iter()
                .filter(|r| !r.6)
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .map(|r| (r.0, format!("{} {}", r.4, r.5))),
            last10: mean(&wpms[n.saturating_sub(10)..]),
            previous10: mean(&wpms[n.saturating_sub(20)..n.saturating_sub(10)]),
            accuracy10: mean(&accs),
            ..Default::default()
        };
        let mut streak = 0;
        for (wpm, _, predicted, hill, _, _, boss) in &runs {
            if !*hill || *predicted <= 0.0 {
                continue;
            }
            // Bosses keep their own record and don't touch the everyday one.
            if *boss {
                match crate::voice::outcome(*wpm, *predicted) {
                    std::cmp::Ordering::Greater => life.boss_wins += 1,
                    std::cmp::Ordering::Less => life.boss_losses += 1,
                    std::cmp::Ordering::Equal => {}
                }
                continue;
            }
            match crate::voice::outcome(*wpm, *predicted) {
                std::cmp::Ordering::Greater => {
                    life.wins += 1;
                    streak += 1;
                    life.longest_streak = life.longest_streak.max(streak);
                }
                std::cmp::Ordering::Less => {
                    life.losses += 1;
                    streak = 0;
                }
                std::cmp::Ordering::Equal => streak = 0,
            }
        }
        Ok(life)
    }

    /// The earliest run before `before_id` (exclusive) whose text contains `word` as a whole
    /// word, ignoring case and punctuation.
    pub fn runs_with(&self, word: &str, before_id: i64) -> Result<Vec<i64>> {
        let mut stmt =
            self.conn.prepare("SELECT id, text FROM runs WHERE id < ?1 AND instr(lower(text), ?2) > 0 ORDER BY id")?;
        let rows: Vec<(i64, String)> =
            stmt.query_map(params![before_id, word], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        Ok(rows
            .into_iter()
            .filter(|(_, text)| text.split(' ').any(|w| crate::generator::bare(w) == word))
            .map(|(id, _)| id)
            .collect())
    }

    pub fn past_run(&self, id: i64) -> Result<PastRun> {
        let started_at = self.conn.query_row("SELECT started_at FROM runs WHERE id = ?1", [id], |r| r.get(0))?;
        let log = self.load_runs("SELECT id, text, wpm, hill FROM runs WHERE id = ?1", id)?.remove(0);
        Ok(PastRun { id, started_at, log })
    }

    pub fn last_run_id(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COALESCE(MAX(id), 0) FROM runs", [], |r| r.get(0))?)
    }

    pub fn setting(&self, key: &str) -> Option<String> {
        self.conn.query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0)).optional().ok().flatten()
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [key, value],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_round_trip() {
        let dir = std::env::temp_dir().join(format!("thehill-store-{}", std::process::id()));
        let mut store = Store::open(&dir).unwrap();
        let events = vec![
            Event { t: 0.0, idx: 0, expected: Some('a'), key: Key::Char('a'), correct: true },
            Event { t: 50.0, idx: 1, expected: None, key: Key::Backspace, correct: false },
        ];
        let run = RunRecord {
            mode: "words",
            amount: 10,
            punctuation: false,
            numbers: false,
            opponent: None,
            text: "a b",
            wpm: 100.0,
            raw: 100.0,
            accuracy: 1.0,
            predicted: 90.0,
            hill: true,
            boss: false,
            model_predicted: 90.0,
        };
        store.save_run(&run, &events).unwrap();
        let logs = store.recent_runs(5).unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].events, events);
        assert_eq!(store.run_count().unwrap(), 1);
        let life = store.lifetime().unwrap();
        assert_eq!((life.runs, life.keystrokes, life.wins, life.losses), (1, 1, 1, 0));
        assert!((life.seconds - 0.05).abs() < 1e-9);
        store.set_setting("mode", "time").unwrap();
        store.set_setting("mode", "words").unwrap();
        assert_eq!(store.setting("mode").as_deref(), Some("words"));

        let failed = store.add_checkpoint(10, &[], 100.0, |_| Err(std::io::Error::other("disk full")));
        assert!(failed.is_err());
        assert!(store.checkpoints().unwrap().is_empty(), "no row without a model file");
        let id = store.add_checkpoint(10, &["abc".into()], 100.0, |p| std::fs::write(p, b"m")).unwrap();
        assert!(store.checkpoint_path(id).exists());
        assert_eq!(store.checkpoints().unwrap().len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
