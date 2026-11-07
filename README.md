# The Hill

A typing trainer in the spirit of monkeytype, with one difficulty: a small neural network that learns which keystrokes slow you down and feeds you text built from them. Fix a weakness and it moves on to the next one.

Native Rust + [GPUI](https://gpui.rs). Linux (Wayland, X11) and macOS.

## Build and run

Linux (Arch):

```sh
sudo pacman -S --needed rust vulkan-icd-loader libxkbcommon wayland libxcb fontconfig
cargo run --release
```

Other distros need the same: Vulkan loader + driver, xkbcommon, wayland, xcb and fontconfig development packages.

macOS: install the Xcode command line tools (`xcode-select --install`) and Rust, then `cargo run --release`.

To install on Linux with a launcher entry:

```sh
cargo install --path . --locked
sed "s|/home/nicholai|$HOME|" assets/thehill.desktop > ~/.local/share/applications/thehill.desktop
install -Dm644 assets/icon.svg ~/.local/share/icons/hicolor/scalable/apps/thehill.svg
```

The text font is JetBrains Mono if it's installed, then DejaVu Sans Mono or Menlo.

## Using it

- Modes: `words 10 / 50`, `time 10 / 15 / 30`. Toggles for punctuation and numbers. Click them in the top bar.
- The run starts on your first keystroke. `tab` gives new text, `esc` restarts the same text, `ctrl+backspace` (or `alt+backspace`) clears the word.
- The prediction is committed before you type and shown above the text; the run is judged against exactly that number (within 3% is a draw).
- The results screen graphs your wpm through the run against the hill's committed pace (dashed) and the ghost, marks your mistakes, and names the word you lost the most time on.
- Boss runs: every 10th run against the live hill is 25 words built from its worst trigrams (four, or more if too few words hold them). It guards the checkpoint that follows; new text or a retry doesn't skip it. Bosses keep their own record.
- Ghost (`ctrl+g`): a faint second caret typing the same text at the pace of the checkpoint frozen about 50 runs ago, i.e. you back then. Its score shows as "past you".
- Profiles (`ctrl+p`, or click the profile name in the top bar): each has its own runs, model and checkpoints. `n` starts a new one from scratch and asks how many chars of context its model reads (3, 5, 8 or 12; the standard model reads 3). `r` renames, `d` twice deletes (moved to `trash/` in the data folder, so it can be recovered). The original data is the main profile and can't be deleted.
- The hill sometimes brings up a word from this run that you also typed at least 20 runs ago, when your time per letter on it has changed by 15% or more.
- `ctrl+s` (or `s` on the results screen) opens the slope: every checkpoint, plotted by how fast it thinks you type. Pick one with the arrow keys and `enter` to play against it. Its text targets the weaknesses you had back then, and the results show how far past its prediction you've moved.

## How the model works

Every keypress is logged to SQLite with its timestamp, the text position it aimed at, and whether it was right. That log is the source of truth. The model is derived from it and can be rebuilt any time with `thehill --retrain`.

The network sees one keystroke at a time: the two chars before it, the char itself (as learned 16-d embeddings), whether it needs shift, and where it sits in its word. Two dense layers of 64 units feed two heads: the log interval in ms, and the probability that you get it wrong. A keystroke's cost is `scale × interval + p(error) × E`. After each training pass, `scale` (mean over typical interval) and `E` (what an error costs you in scored time) are fitted so that predicted wpm matches the wpm you actually scored over recent runs; `thehill --calibrate` shows the fit run by run. The embeddings are shared across all contexts, so a slow `ght` makes the model expect `ghts` and `ought` to be slow too before you've typed them.

After every run it trains for a few epochs on the last 300 runs on a background thread. Recent runs weigh more (half-life of 40 runs), so weaknesses you've fixed fade out.

To build each run:

- 80% of words come from the 1500 words with the highest mean keystroke cost. Each one is rescored for the actual transition out of the previous word (`end of word → space → start of word`) and sampled with a softmax, not just taken as the max.
- 20% explore: words full of transitions the model has seen little of.
- Difficulty has to come from transitions, not familiarity tricks. A word family (words agreeing on their first four letters, or a three-letter word and its extensions: yeet/yeeting, hydro/hydra) appears at most once per run, and the last 150 distinct words you typed rest, families included, before they come back.
- With punctuation or numbers on, a few random decorations are drawn for each word and the costliest is kept.

Every 10 runs the live weights are frozen as a checkpoint. The label shows the date, the run count, and the four common trigrams it found hardest (`_` is a space).

The first 3 runs use plain common words while the model gets its first data.

## Data

`~/.local/share/thehill` on Linux, `~/Library/Application Support/thehill` on macOS. Other profiles live in `profiles/` inside it. Data from the earlier name (Sisyphus) is copied over on first launch. Override with `THEHILL_DATA_DIR`.

- `thehill.db`: runs, keystrokes, checkpoints, settings
- `live.bin`: current model
- `checkpoints/<id>.bin`: frozen models

## CLI

```sh
thehill --stats                         # runs, checkpoints, weakest transitions, hardest words
thehill --retrain                       # rebuild the live model from the whole log
thehill --simulate 60 --slow th,er      # synthetic typist, headless; uses a temp dir unless THEHILL_DATA_DIR is set
```

## Word list

`assets/words.txt` holds the 20,000 most frequent English words per [wordfreq](https://github.com/rspeer/wordfreq) that pass the SCOWL-based en_US Hunspell dictionary (lowercase forms only, so names drop out), minus a profanity list. `tools/build_words.py` regenerates it. Sources and licenses are in `assets/WORDS_SOURCE.md`.
