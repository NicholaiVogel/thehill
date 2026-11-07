//! Profiles: separate players or experiments, each with its own runs, model and checkpoints.
//!
//! The original data directory is the main profile, left exactly where it was. Others live
//! in `profiles/<id>` beside it. Deleting a profile moves it to `trash/`, so a regretted
//! delete can be undone by hand. The main profile can't be deleted.

use crate::model::DEFAULT_CTX;
use crate::storage::Store;
use anyhow::{Result, bail};
use std::path::PathBuf;

pub const MAIN: &str = "main";
/// Settings a new profile starts with, copied from the one you're in: how the app looks and
/// what you play, nothing the hill has learned.
const CARRIED: [&str; 7] = ["theme", "zoom", "mode", "punctuation", "numbers", "hill", "ghost"];

pub struct Profile {
    pub id: String,
    pub name: String,
    /// Chars of context its model sees.
    pub context: usize,
    pub runs: i64,
}

#[cfg(test)]
thread_local! {
    static TEST_ROOT: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

fn data_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(root) = TEST_ROOT.with(|r| r.borrow().clone()) {
        return root;
    }
    crate::storage::data_dir()
}

pub fn dir_of(id: &str) -> PathBuf {
    if id == MAIN { data_dir() } else { data_dir().join("profiles").join(id) }
}

fn active_file() -> PathBuf {
    data_dir().join("active-profile")
}

/// The profile in use; the main one if the recorded one is gone.
pub fn active() -> String {
    let id = std::fs::read_to_string(active_file()).unwrap_or_default().trim().to_string();
    if valid_id(&id) && id != MAIN && dir_of(&id).join("thehill.db").exists() { id } else { MAIN.into() }
}

pub fn set_active(id: &str) -> Result<()> {
    std::fs::create_dir_all(data_dir())?;
    std::fs::write(active_file(), id)?;
    Ok(())
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

/// The context length a profile's model is built with.
pub fn context(store: &Store) -> usize {
    store.setting("context").and_then(|c| c.parse().ok()).unwrap_or(DEFAULT_CTX)
}

pub fn name(store: &Store, id: &str) -> String {
    store.setting("profile_name").filter(|n| !n.trim().is_empty()).unwrap_or_else(|| id.to_string())
}

fn load(id: &str) -> Result<Profile> {
    let store = if id == MAIN { Store::open(&dir_of(id))? } else { Store::open_existing(&dir_of(id))? };
    Ok(Profile { id: id.into(), name: name(&store, id), context: context(&store), runs: store.run_count()? })
}

/// Main first, then the others oldest first, and a message naming any that couldn't be read.
pub fn list() -> (Vec<Profile>, Option<String>) {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("profiles"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("thehill.db").exists())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|id| valid_id(id) && id != MAIN)
        .collect();
    ids.sort();
    let (mut found, mut broken) = (Vec::new(), Vec::new());
    for id in std::iter::once(MAIN.to_string()).chain(ids) {
        match load(&id) {
            Ok(p) => found.push(p),
            Err(_) => broken.push(id),
        }
    }
    let error = (!broken.is_empty()).then(|| format!("couldn't read {}", broken.join(", ")));
    (found, error)
}

/// A fresh profile whose model sees `context` chars, starting with `from`'s look and mode.
pub fn create(context: usize, from: &Store) -> Result<Profile> {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
    let id = format!("p{stamp}");
    let dir = dir_of(&id);
    std::fs::create_dir_all(&dir)?;
    let store = Store::open(&dir)?;
    for key in CARRIED {
        if let Some(v) = from.setting(key) {
            store.set_setting(key, &v)?;
        }
    }
    store.set_setting("context", &context.to_string())?;
    let base = format!("context {context}");
    let taken: Vec<String> = list().0.into_iter().map(|p| p.name).collect();
    let name = (1..)
        .map(|n| if n == 1 { base.clone() } else { format!("{base} ({n})") })
        .find(|n| !taken.contains(n))
        .unwrap();
    store.set_setting("profile_name", &name)?;
    drop(store);
    load(&id)
}

pub fn rename(id: &str, name: &str) -> Result<()> {
    let name = name.trim();
    if name.is_empty() {
        bail!("a profile needs a name");
    }
    if !valid_id(id) {
        bail!("no such profile");
    }
    let dir = dir_of(id);
    let store = if id == MAIN { Store::open(&dir)? } else { Store::open_existing(&dir)? };
    store.set_setting("profile_name", name)
}

/// Moves a profile to `trash/`. The main profile stays.
pub fn delete(id: &str) -> Result<()> {
    if id == MAIN || !valid_id(id) {
        bail!("the main profile can't be deleted");
    }
    let trash = data_dir().join("trash");
    std::fs::create_dir_all(&trash)?;
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    std::fs::rename(dir_of(id), trash.join(format!("{id}-{stamp}")))?;
    if active() == id {
        set_active(MAIN)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_created_renamed_switched_and_trashed() {
        let root = std::env::temp_dir().join(format!("thehill-profiles-{}", std::process::id()));
        TEST_ROOT.with(|r| *r.borrow_mut() = Some(root.clone()));
        let main = Store::open(&root).unwrap();
        main.set_setting("theme", "light").unwrap();
        main.set_setting("voice", "secret").unwrap();

        assert_eq!(context(&main), DEFAULT_CTX);
        let p = create(8, &main).unwrap();
        assert_eq!((p.name.as_str(), p.context, p.runs), ("context 8", 8, 0));
        let store = Store::open(&dir_of(&p.id)).unwrap();
        assert_eq!(store.setting("theme").as_deref(), Some("light"), "look carries over");
        assert_eq!(store.setting("voice"), None, "what the hill knows doesn't");
        assert_eq!(create(8, &main).unwrap().name, "context 8 (2)");

        set_active(&p.id).unwrap();
        assert_eq!(active(), p.id);
        rename(&p.id, "  long one ").unwrap();
        assert!(rename(&p.id, " ").is_err());
        assert!(rename("", "x").is_err() && rename("pnope", "x").is_err(), "no profile is created by renaming");
        assert!(!dir_of("pnope").exists());
        assert_eq!(list().0.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["main", "long one", "context 8 (2)"]);

        assert!(delete(MAIN).is_err());
        delete(&p.id).unwrap();
        assert_eq!(active(), MAIN, "deleting the active profile falls back to main");
        assert_eq!(list().0.len(), 2);
        // A broken profile is reported, not allowed to hide the others.
        let broken = dir_of("pbroken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("thehill.db"), b"not a database").unwrap();
        let (found, error) = list();
        assert_eq!(found.len(), 2);
        assert!(error.unwrap().contains("pbroken"));
        assert_eq!(std::fs::read_dir(root.join("trash")).unwrap().count(), 1, "kept in the trash");
        std::fs::remove_dir_all(root).unwrap();
    }
}
