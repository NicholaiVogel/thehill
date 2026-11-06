//! The window: config bar, typing area, results, the slope, and the hill's voice.

use crate::learn::{self, CHECKPOINT_EVERY};
use crate::session::{Finished, Mode, Run, Session, train};
use crate::storage::{Checkpoint, Lifetime};
use crate::typing::{Key, Test};
use crate::voice::Heckle;
use gpui::{
    App, Application, Bounds, ClickEvent, Context, FocusHandle, Font, HighlightStyle, Hsla, KeyDownEvent, PathBuilder,
    Pixels, Point, SharedString, StyledText, Window, WindowBounds, WindowOptions, canvas, div, font, point, prelude::*,
    px, rgb, size,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Minimal like monkeytype, but neutral: grays and off-white with no brand colour. Red
/// is kept for errors only, so it always means something.
struct Palette {
    bg: u32,
    untyped: u32,
    correct: u32,
    error: u32,
    extra: u32,
    /// Active settings, caret, headline numbers: the brightest text, not a hue.
    accent: u32,
    key_bg: u32,
}

const DARK: Palette = Palette {
    bg: 0x232427,
    untyped: 0x5c5e63,
    correct: 0xcfcdc7,
    error: 0xd0575a,
    extra: 0x7a3236,
    accent: 0xf2f0ea,
    key_bg: 0x2d2e32,
};

const LIGHT: Palette = Palette {
    bg: 0xeceae5,
    untyped: 0xa29f99,
    correct: 0x3a3b3e,
    error: 0xc23b3f,
    extra: 0xe2a4a4,
    accent: 0x111214,
    key_bg: 0xdedcd6,
};

/// One window, one theme: a process-wide switch keeps colour lookups plain.
static LIGHT_MODE: AtomicBool = AtomicBool::new(false);

fn pal() -> &'static Palette {
    if LIGHT_MODE.load(Ordering::Relaxed) { &LIGHT } else { &DARK }
}

const TEXT_SIZE: f32 = 28.0;
const LINE_HEIGHT: f32 = TEXT_SIZE * 1.6;
const UI_SIZE: f32 = 14.0;
const VOICE_SIZE: f32 = 16.0;
const MAX_WIDTH: f32 = 900.0;
/// Roughly how long the caret takes to catch up with a keystroke. It's a critically damped
/// spring, so at speed it glides continuously instead of restarting a tween on every key.
const CARET_SMOOTH_S: f32 = 0.045;
/// Height of the macOS titlebar the content extends under (traffic lights sit in it).
const MAC_TITLEBAR: f32 = if cfg!(target_os = "macos") { 28.0 } else { 0.0 };
const CHART_HEIGHT: f32 = 200.0;
const FONTS: [&str; 5] = ["JetBrains Mono", "JetBrainsMono Nerd Font Mono", "DejaVu Sans Mono", "Menlo", "Monaco"];

const ZOOM_STEP: f32 = 0.1;
const ZOOM_RANGE: (f32, f32) = (0.6, 2.0);

/// Typewriter speed for the hill's lines, and the pause between lines.
const MS_PER_CHAR: f32 = 24.0;
const LINE_PAUSE_CHARS: usize = 18;
/// A heckle shows this long, then fades.
const HECKLE_MS: u128 = 2500;
const FADE_MS: f32 = 600.0;
const HECKLE_GAP: Duration = Duration::from_secs(4);
const HECKLES_PER_RUN: u32 = 4;

/// Snaps to the zoom steps and range; anything non-finite falls back to 100%.
fn normalize_zoom(zoom: f32) -> f32 {
    if !zoom.is_finite() {
        return 1.0;
    }
    ((zoom / ZOOM_STEP).round() * ZOOM_STEP).clamp(ZOOM_RANGE.0, ZOOM_RANGE.1)
}

fn color(hex: u32) -> Hsla {
    rgb(hex).into()
}

enum Screen {
    Typing,
    /// `shown` holds when each of the hill's lines was added; late lines type out from then.
    Results {
        done: Finished,
        shown: Vec<Instant>,
    },
    Stats(Stats),
}

/// The stats screen: lifetime numbers, the checkpoint slope, and the selected model's view of you.
struct Stats {
    checkpoints: Vec<Checkpoint>,
    live_wpm: Option<f64>,
    lifetime: Lifetime,
    /// Index into `checkpoints`; one past the end is the live model.
    selected: usize,
    tags: Vec<String>,
    heat: Vec<(char, Option<f32>)>,
}

struct Caret {
    /// Where it's drawn; y always equals the target's (line changes jump).
    x: f32,
    velocity: f32,
    to: Point<Pixels>,
    /// When the target last changed, for the idle blink.
    since: Instant,
    last_frame: Instant,
}

impl Caret {
    fn new(now: Instant) -> Caret {
        Caret { x: 0.0, velocity: 0.0, to: Point::default(), since: now, last_frame: now }
    }

    fn retarget(&mut self, to: Point<Pixels>, now: Instant) {
        if to.y != self.to.y {
            self.x = f32::from(to.x);
            self.velocity = 0.0;
        }
        self.to = to;
        self.since = now;
    }

    /// Advances the spring to `now`; returns the position and whether it's still moving.
    fn step(&mut self, now: Instant) -> (Point<Pixels>, bool) {
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        let target = f32::from(self.to.x);
        // Exact critically damped spring step (as in Unity's SmoothDamp).
        let omega = 2.0 / CARET_SMOOTH_S;
        let k = omega * dt;
        let decay = 1.0 / (1.0 + k + 0.48 * k * k + 0.235 * k * k * k);
        let offset = self.x - target;
        let temp = (self.velocity + omega * offset) * dt;
        self.velocity = (self.velocity - omega * temp) * decay;
        self.x = target + (offset + temp) * decay;
        let moving = (self.x - target).abs() > 0.1 || self.velocity.abs() > 1.0;
        if !moving {
            self.x = target;
            self.velocity = 0.0;
        }
        (point(px(self.x), self.to.y), moving)
    }
}

/// One line from the hill under the typing area: typed out, then optionally faded away.
struct Speech {
    text: String,
    since: Instant,
    fade_at: Option<Instant>,
    /// Announcements can't be talked over by heckles until they've been read.
    important: bool,
}

impl Speech {
    fn new(text: String) -> Speech {
        Speech { text, since: Instant::now(), fade_at: None, important: false }
    }

    fn opacity(&self, now: Instant) -> f32 {
        self.fade_at.map_or(1.0, |f| 1.0 - (now.saturating_duration_since(f).as_secs_f32() * 1000.0 / FADE_MS).min(1.0))
    }
}

/// How many chars of typewriter text are revealed after `since`.
fn revealed(since: Instant, now: Instant) -> usize {
    (now.duration_since(since).as_secs_f32() * 1000.0 / MS_PER_CHAR) as usize
}

struct TheHill {
    session: Session,
    run: Run,
    screen: Screen,
    focus: FocusHandle,
    started: Option<Instant>,
    caret: Caret,
    ghost_caret: Caret,
    font: SharedString,
    zoom: f32,
    training: bool,
    train_again: bool,
    speech: Option<Speech>,
    /// Said at the next opportunity: a checkpoint announced mid-run.
    pending: Option<String>,
    last_heckle: Option<Instant>,
    heckles: u32,
}

pub fn run(session: Session) {
    Application::new().run(move |cx: &mut App| {
        let names = cx.text_system().all_font_names();
        let family = FONTS.iter().find(|f| names.iter().any(|n| n == *f)).copied().unwrap_or("monospace");
        let bounds = Bounds::centered(None, size(px(1100.0), px(700.0)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            app_id: Some("thehill".into()),
            // On macOS the content runs under a transparent titlebar, traffic lights inside the
            // window. Linux keeps the compositor's decorations.
            titlebar: Some(gpui::TitlebarOptions {
                title: Some("The Hill".into()),
                appears_transparent: cfg!(target_os = "macos"),
                traffic_light_position: cfg!(target_os = "macos").then(|| point(px(14.0), px(14.0))),
            }),
            ..Default::default()
        };
        cx.open_window(options, |window, cx| {
            cx.new(|cx| {
                let focus = cx.focus_handle();
                focus.focus(window);
                let mut session = session;
                let zoom = session.store.setting("zoom").and_then(|z| z.parse().ok()).map_or(1.0, normalize_zoom);
                LIGHT_MODE.store(session.store.setting("theme").as_deref() == Some("light"), Ordering::Relaxed);
                let run = session.new_run().expect("could not start a run");
                let speech = Some(Speech::new(run.preview.clone()));
                let app = TheHill {
                    session,
                    run,
                    screen: Screen::Typing,
                    focus,
                    started: None,
                    caret: Caret::new(Instant::now()),
                    ghost_caret: Caret::new(Instant::now()),
                    font: family.into(),
                    zoom,
                    training: false,
                    train_again: false,
                    speech,
                    pending: None,
                    last_heckle: None,
                    heckles: 0,
                };
                cx.spawn(async move |this, cx| {
                    loop {
                        cx.background_executor().timer(Duration::from_millis(50)).await;
                        if this.update(cx, |this: &mut TheHill, cx| this.tick(cx)).is_err() {
                            break;
                        }
                    }
                })
                .detach();
                app
            })
        })
        .expect("could not open the window");
        cx.activate(true);
    });
}

impl TheHill {
    /// A design size in pixels at the current zoom.
    fn z(&self, v: f32) -> Pixels {
        px(v * self.zoom)
    }

    fn elapsed_ms(&self) -> f64 {
        self.started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0)
    }

    fn running(&self) -> bool {
        matches!(self.screen, Screen::Typing) && self.started.is_some()
    }

    fn out_of_time(&self) -> bool {
        matches!(self.run.mode, Mode::Time(limit) if self.running() && self.elapsed_ms() >= limit as f64 * 1000.0)
    }

    fn tick(&mut self, cx: &mut Context<Self>) {
        if self.out_of_time() {
            return self.finish(cx);
        }
        if let (true, Mode::Time(limit)) = (self.running(), self.run.mode) {
            let left = limit as f64 - self.elapsed_ms() / 1000.0;
            // Only while "three seconds" is still roughly true.
            if left <= 3.0 && left > 2.0 && limit > 5 {
                self.heckle(Heckle::LastSeconds);
            }
        }
        let now = Instant::now();
        if let Some(s) = &mut self.speech {
            let shown_for = now.duration_since(s.since).as_millis();
            let typed_ms = (s.text.len() as f32 * MS_PER_CHAR) as u128;
            if s.fade_at.is_none() && self.started.is_some() && shown_for > typed_ms + HECKLE_MS {
                s.fade_at = Some(now);
            }
            if s.opacity(now) <= 0.0 {
                self.speech = None;
            }
        }
        // Repaint for the countdown, the caret blink and the hill's lines.
        if !matches!(self.screen, Screen::Stats(_)) {
            cx.notify();
        }
    }

    fn new_run(&mut self, cx: &mut Context<Self>) {
        match self.session.new_run() {
            Ok(run) => self.run = run,
            Err(e) => eprintln!("thehill: could not generate a run: {e:#}"),
        }
        let line = self.pending.take().unwrap_or_else(|| self.run.preview.clone());
        self.speech = Some(Speech::new(line));
        self.reset(cx);
    }

    fn restart_same_text(&mut self, cx: &mut Context<Self>) {
        let words = self.run.test.words.iter().map(|w| w.iter().collect()).collect();
        self.run.test = Test::new(words);
        self.speech = None;
        self.reset(cx);
    }

    fn reset(&mut self, cx: &mut Context<Self>) {
        self.started = None;
        self.ghost_caret = Caret::new(Instant::now());
        self.heckles = 0;
        self.last_heckle = None;
        self.screen = Screen::Typing;
        cx.notify();
    }

    fn finish(&mut self, cx: &mut Context<Self>) {
        self.run.test.finished = true;
        match self.session.finish(&self.run) {
            Ok(mut done) => {
                done.lines.extend(self.pending.take());
                let shown = vec![Instant::now(); done.lines.len()];
                self.screen = Screen::Results { done, shown };
            }
            Err(e) => {
                eprintln!("thehill: could not save the run: {e:#}");
                return self.new_run(cx);
            }
        }
        self.started = None;
        self.speech = None;
        if self.training {
            self.train_again = true;
        } else {
            self.train(cx);
        }
        cx.notify();
    }

    /// Trains on a background thread; runs again if another run finished meanwhile.
    fn train(&mut self, cx: &mut Context<Self>) {
        self.training = true;
        let dir = self.session.dir.clone();
        let base = self.session.training_base();
        cx.spawn(async move |this, cx| {
            let result = cx.background_executor().spawn(async move { train(&dir, base) }).await;
            this.update(cx, |this, cx| {
                match result {
                    Ok((model, costs, announcement)) => {
                        this.session.install(model, costs);
                        if let Some(line) = announcement.filter(|_| this.session.hill) {
                            this.announce(line);
                        }
                    }
                    Err(e) => eprintln!("thehill: training failed: {e:#}"),
                }
                this.training = false;
                if std::mem::take(&mut this.train_again) {
                    this.train(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Says `line` now if there's room for it, otherwise at the next results or run.
    fn announce(&mut self, line: String) {
        match &mut self.screen {
            Screen::Results { done, shown } => {
                done.lines.push(line);
                shown.push(Instant::now());
            }
            Screen::Typing if self.started.is_none() => {
                self.speech = Some(Speech { important: true, ..Speech::new(line) })
            }
            _ => self.pending = Some(line),
        }
    }

    fn heckle(&mut self, h: Heckle) {
        let quiet = self.last_heckle.is_some_and(|t| t.elapsed() < HECKLE_GAP);
        let last_seconds = matches!(h, Heckle::LastSeconds);
        if quiet || self.heckles >= HECKLES_PER_RUN + u32::from(last_seconds) {
            return;
        }
        // The countdown line is said once, and only if nothing else is on screen.
        if last_seconds && self.speech.is_some() {
            return;
        }
        if self.speech.as_ref().is_some_and(|s| s.important) {
            return;
        }
        self.heckles += 1;
        self.last_heckle = Some(Instant::now());
        self.speech = Some(Speech::new(self.session.voice.heckle(&h)));
        self.session.save_voice();
    }

    /// Looks at the keypress just made for something worth commenting on.
    fn react(&mut self) {
        let events = &self.run.test.events;
        let Some(last) = events.last() else { return };
        if last.key == Key::Backspace || last.key == Key::ClearWord {
            return;
        }
        if !last.correct && last.expected.is_some() && last.idx >= 2 {
            let text: Vec<char> = self.run.test.text().chars().collect();
            let tag: String = text[last.idx - 2..=last.idx].iter().map(|&c| if c == ' ' { '_' } else { c }).collect();
            if self.run.weak.contains(&tag) {
                return self.heckle(Heckle::Weakness(tag));
            }
            let recent = events.iter().rev().take_while(|e| last.t - e.t < 2000.0);
            if recent.filter(|e| matches!(e.key, Key::Char(_)) && !e.correct).count() >= 3 {
                return self.heckle(Heckle::Burst);
            }
        }
        if events.len() > 15 && last.correct {
            let mean = (last.t - events[0].t) / (events.len() - 1) as f64;
            let gap = last.t - events[events.len() - 2].t;
            if gap > (3.0 * mean).max(600.0) {
                self.heckle(Heckle::Hesitation);
            }
        }
    }

    fn open_stats(&mut self, cx: &mut Context<Self>) {
        let store = &self.session.store;
        let checkpoints = store.checkpoints().unwrap_or_default();
        let selected = match self.session.opponent_id {
            Some(id) => checkpoints.iter().position(|c| c.id == id).unwrap_or(checkpoints.len()),
            None => checkpoints.len(),
        };
        self.started = None;
        self.screen = Screen::Stats(Stats {
            live_wpm: self.session.live.as_ref().map(learn::reference_wpm),
            lifetime: store.lifetime().unwrap_or_default(),
            checkpoints,
            selected,
            tags: Vec::new(),
            heat: Vec::new(),
        });
        self.select(selected, cx);
    }

    /// Selects a checkpoint (or live, one past the end) and loads its view of you.
    fn select(&mut self, i: usize, cx: &mut Context<Self>) {
        let Screen::Stats(st) = &mut self.screen else { return };
        st.selected = i.min(st.checkpoints.len());
        let model = match st.checkpoints.get(st.selected) {
            Some(c) => learn::load_checkpoint(&self.session.store, c.id).ok().flatten(),
            None => self.session.live.clone(),
        };
        st.tags = match (st.checkpoints.get(st.selected), &model) {
            (Some(c), _) => c.tags.clone(),
            (None, Some(m)) => learn::weaknesses(m, 4),
            (None, None) => Vec::new(),
        };
        st.heat = model.as_ref().map(learn::key_costs).unwrap_or_default();
        cx.notify();
    }

    fn choose_opponent(&mut self, id: Option<i64>, cx: &mut Context<Self>) {
        if let Err(e) = self.session.set_opponent(id).and_then(|_| self.session.save_settings()) {
            eprintln!("thehill: could not switch opponent: {e:#}");
        }
        self.new_run(cx);
    }

    fn change_settings(&mut self, cx: &mut Context<Self>, change: impl FnOnce(&mut Session)) {
        // The bar is invisible during a run but still receives clicks.
        if self.running() {
            return;
        }
        change(&mut self.session);
        if let Err(e) = self.session.save_settings() {
            eprintln!("thehill: could not save settings: {e:#}");
        }
        self.new_run(cx);
    }

    fn toggle_theme(&mut self, cx: &mut Context<Self>) {
        let light = !LIGHT_MODE.load(Ordering::Relaxed);
        LIGHT_MODE.store(light, Ordering::Relaxed);
        if let Err(e) = self.session.store.set_setting("theme", if light { "light" } else { "dark" }) {
            eprintln!("thehill: could not save theme: {e:#}");
        }
        cx.notify();
    }

    fn set_zoom(&mut self, zoom: f32, cx: &mut Context<Self>) {
        self.zoom = normalize_zoom(zoom);
        if let Err(e) = self.session.store.set_setting("zoom", &format!("{:.1}", self.zoom)) {
            eprintln!("thehill: could not save zoom: {e:#}");
        }
        cx.notify();
    }

    fn on_key(&mut self, ev: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let ks = &ev.keystroke;
        let m = ks.modifiers;
        if m.control {
            match ks.key.as_str() {
                "s" => return self.open_stats(cx),
                "h" if !self.running() => return self.change_settings(cx, |s| s.hill ^= true),
                "t" => return self.toggle_theme(cx),
                "g" if !self.running() => return self.change_settings(cx, |s| s.ghost ^= true),
                "=" | "+" => return self.set_zoom(self.zoom + ZOOM_STEP, cx),
                "-" => return self.set_zoom(self.zoom - ZOOM_STEP, cx),
                "0" => return self.set_zoom(1.0, cx),
                _ => {}
            }
        }
        match &mut self.screen {
            Screen::Stats(st) => {
                let (selected, id) = (st.selected, st.checkpoints.get(st.selected).map(|c| c.id));
                match ks.key.as_str() {
                    "left" => return self.select(selected.saturating_sub(1), cx),
                    "right" => return self.select(selected + 1, cx),
                    "enter" => return self.choose_opponent(id, cx),
                    "escape" => return self.new_run(cx),
                    _ => {}
                }
            }
            Screen::Results { .. } => match ks.key.as_str() {
                "enter" | "tab" => return self.new_run(cx),
                "escape" => return self.restart_same_text(cx),
                "s" => return self.open_stats(cx),
                _ => {}
            },
            Screen::Typing => {
                // Keys that arrive after the deadline but before the next tick don't count.
                if self.out_of_time() {
                    return self.finish(cx);
                }
                let t = self.elapsed_ms();
                let test = &mut self.run.test;
                match ks.key.as_str() {
                    "tab" => return self.new_run(cx),
                    "escape" => return self.restart_same_text(cx),
                    "backspace" if m.control || m.alt => test.clear_word(t),
                    "backspace" => test.backspace(t),
                    "space" => test.type_char(' ', t),
                    _ if m.control || m.platform || m.alt || m.function => return,
                    _ => match &ks.key_char {
                        Some(s) => s.chars().for_each(|c| test.type_char(c, t)),
                        None => return,
                    },
                }
                if self.started.is_none() && test.started() {
                    self.started = Some(Instant::now());
                }
                self.run.top_up();
                if self.run.test.finished {
                    return self.finish(cx);
                }
                self.react();
            }
        }
        window.prevent_default();
        cx.stop_propagation();
        cx.notify();
    }

    fn render_bar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let item = |id: &'static str, label: String, active: bool| {
            div()
                .id(id)
                .cursor_pointer()
                .text_color(color(if active { pal().accent } else { pal().untyped }))
                .hover(|s| s.text_color(color(if active { pal().accent } else { pal().correct })))
                .child(label)
        };
        let divider = || div().w(px(1.0)).h(self.z(UI_SIZE)).bg(color(pal().untyped));
        let options = self.session.options;
        let mode = self.session.mode;
        let mode_items = |words: bool| {
            let mut row = div().flex().gap(self.z(16.0));
            for (i, m) in Mode::ALL.into_iter().enumerate() {
                let (is_words, n) = match m {
                    Mode::Words(n) => (true, n),
                    Mode::Time(n) => (false, n),
                };
                if is_words == words {
                    let ids = ["m0", "m1", "m2", "m3", "m4"];
                    row = row.child(item(ids[i], n.to_string(), mode == m).on_click(
                        cx.listener(move |this, _: &ClickEvent, _, cx| this.change_settings(cx, |s| s.mode = m)),
                    ));
                }
            }
            row
        };
        let opponent = match self.session.opponent_id {
            None => "vs live".to_string(),
            Some(id) => self
                .session
                .store
                .checkpoints()
                .ok()
                .and_then(|cs| cs.into_iter().find(|c| c.id == id))
                .map_or("vs checkpoint".into(), |c| format!("vs {}", c.label())),
        };
        div()
            .flex()
            .flex_wrap()
            .justify_center()
            .items_center()
            .gap(self.z(16.0))
            .px(self.z(24.0))
            .text_size(self.z(UI_SIZE))
            .opacity(if self.running() { 0.0 } else { 1.0 })
            .child(
                item("hill", "the hill".into(), self.session.hill)
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.change_settings(cx, |s| s.hill ^= true))),
            )
            .child(
                item("ghost", "ghost".into(), self.session.ghost)
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.change_settings(cx, |s| s.ghost ^= true))),
            )
            .child(divider())
            .child(item("punctuation", "punctuation".into(), options.punctuation).on_click(
                cx.listener(|this, _: &ClickEvent, _, cx| this.change_settings(cx, |s| s.options.punctuation ^= true)),
            ))
            .child(item("numbers", "numbers".into(), options.numbers).on_click(
                cx.listener(|this, _: &ClickEvent, _, cx| this.change_settings(cx, |s| s.options.numbers ^= true)),
            ))
            .child(divider())
            .child(item("words", "words".into(), matches!(mode, Mode::Words(_))))
            .child(mode_items(true))
            .child(divider())
            .child(item("time", "time".into(), matches!(mode, Mode::Time(_))))
            .child(mode_items(false))
            .child(divider())
            .child(item("opponent", if self.session.hill { opponent } else { "stats".into() }, false).on_click(
                cx.listener(|this, _: &ClickEvent, _, cx| {
                    if !this.running() {
                        this.open_stats(cx)
                    }
                }),
            ))
            .child(divider())
            .child(
                item("theme", if LIGHT_MODE.load(Ordering::Relaxed) { "light" } else { "dark" }.into(), false)
                    .on_click(cx.listener(|this, _: &ClickEvent, _, cx| {
                        if !this.running() {
                            this.toggle_theme(cx)
                        }
                    })),
            )
    }

    fn render_typing(&mut self, width: f32, char_w: f32, window: &mut Window) -> impl IntoElement {
        let line_h = LINE_HEIGHT * self.zoom;
        let cols = (width / char_w).floor().max(10.0) as usize;
        let test = &self.run.test;
        let shown = |i: usize| test.words[i].len().max(test.inputs[i].len());

        // Greedy word wrap on the displayed lengths (extras widen a word).
        let mut lines: Vec<std::ops::Range<usize>> = Vec::new();
        let (mut start, mut used) = (0, 0);
        for i in 0..test.words.len() {
            let need = shown(i) + usize::from(i > start);
            if i > start && used + need > cols {
                lines.push(start..i);
                (start, used) = (i, shown(i));
            } else {
                used += need;
            }
        }
        lines.push(start..test.words.len());
        let current_line = lines.iter().position(|l| l.contains(&test.current)).unwrap_or(0);
        let top = current_line.saturating_sub(1);

        // Where the ghost is: chars typed at its pace so far, as (word, offset in word).
        let ghost_at = self.run.ghost.as_ref().filter(|_| self.started.is_some()).map(|g| {
            let typed = g.pace.partition_point(|&p| p as f64 <= self.elapsed_ms());
            let (mut word, mut start) = (0, 0);
            while word + 1 < test.words.len() && start + test.words[word].len() < typed {
                start += test.words[word].len() + 1;
                word += 1;
            }
            (word, (typed - start).min(test.words[word].len()))
        });
        let mut ghost = None;
        let mut rows = Vec::new();
        let mut caret = Point::default();
        for (row, range) in lines.iter().enumerate().skip(top).take(3) {
            let (mut text, mut highlights, mut col) = (String::new(), Vec::new(), 0);
            for i in range.clone() {
                if i > range.start {
                    text.push(' ');
                    col += 1;
                }
                if i == test.current {
                    caret = point(px((col + test.inputs[i].len()) as f32 * char_w), px((row - top) as f32 * line_h));
                }
                if let Some((_, k)) = ghost_at.filter(|(g, _)| *g == i) {
                    ghost = Some(point(px((col + k) as f32 * char_w), px((row - top) as f32 * line_h)));
                }
                let (word, input) = (&test.words[i], &test.inputs[i]);
                for k in 0..shown(i) {
                    let (c, hex) = match (word.get(k), input.get(k)) {
                        (Some(&w), Some(&typed)) => (w, if typed == w { pal().correct } else { pal().error }),
                        (Some(&w), None) => (w, pal().untyped),
                        (None, Some(&typed)) => (typed, pal().extra),
                        (None, None) => unreachable!(),
                    };
                    let at = text.len();
                    text.push(c);
                    if hex != pal().untyped {
                        highlights
                            .push((at..text.len(), HighlightStyle { color: Some(color(hex)), ..Default::default() }));
                    }
                }
                col += shown(i);
            }
            // The wrap above is authoritative; the text layer must never re-wrap a row, or a
            // word spills onto the row below when glyph rounding makes the line a hair wider.
            rows.push(div().h(px(line_h)).whitespace_nowrap().child(StyledText::new(text).with_highlights(highlights)));
        }

        let now = Instant::now();
        if caret != self.caret.to {
            self.caret.retarget(caret, now);
        }
        let (pos, moving) = self.caret.step(now);
        if moving {
            window.request_animation_frame();
        }
        let ghost = ghost.map(|g| {
            if g != self.ghost_caret.to {
                self.ghost_caret.retarget(g, now);
            }
            // The ghost moves on its own clock, so keep frames coming while it's on screen.
            window.request_animation_frame();
            self.ghost_caret.step(now).0
        });
        let blink_off = self.started.is_none() && now.duration_since(self.caret.since).as_millis() % 1000 >= 500;
        let caret_h = TEXT_SIZE * 1.2 * self.zoom;

        div()
            .relative()
            .w(px((cols as f32 + 0.5) * char_w))
            .h(px(3.0 * line_h))
            .overflow_hidden()
            .text_size(self.z(TEXT_SIZE))
            .line_height(px(line_h))
            .text_color(color(pal().untyped))
            .children(rows)
            .child(
                div()
                    .absolute()
                    .left(pos.x - px(1.0))
                    .top(pos.y + px((line_h - caret_h) / 2.0))
                    .w(self.z(2.0))
                    .h(px(caret_h))
                    .rounded(px(1.0))
                    .bg(color(pal().accent))
                    .opacity(if blink_off { 0.0 } else { 1.0 }),
            )
            .when_some(ghost, |d, g| {
                d.child(
                    div()
                        .absolute()
                        .left(g.x - px(1.0))
                        .top(g.y + px((line_h - caret_h) / 2.0))
                        .w(self.z(2.0))
                        .h(px(caret_h))
                        .rounded(px(1.0))
                        .bg(color(pal().untyped)),
                )
            })
    }

    /// The hill's line under the typing area. The slot keeps its height so nothing jumps.
    fn render_speech(&self, window: &mut Window) -> impl IntoElement {
        let now = Instant::now();
        let (text, opacity) = match &self.speech {
            Some(s) => {
                let n = revealed(s.since, now);
                if n < s.text.len() {
                    window.request_animation_frame();
                }
                (s.text.chars().take(n).collect::<String>(), s.opacity(now))
            }
            None => (String::new(), 0.0),
        };
        div()
            .h(self.z(VOICE_SIZE * 2.0))
            .text_size(self.z(VOICE_SIZE))
            .text_color(color(pal().correct))
            .opacity(opacity)
            .child(text)
    }

    fn counter(&self) -> String {
        if self.started.is_none() {
            // Committed before you type, so the hill can't move the line afterwards.
            return match self.run.predicted {
                p if p > 0.0 && self.run.boss => format!("boss  ·  predicts {p:.0}"),
                p if p > 0.0 => format!("predicts {p:.0}"),
                _ => String::new(),
            };
        }
        match self.run.mode {
            Mode::Time(limit) => ((limit as f64 - self.elapsed_ms() / 1000.0).ceil().max(0.0) as u32).to_string(),
            Mode::Words(n) => format!("{}/{n}", self.run.test.current),
        }
    }

    fn stat(&self, label: &str, value: String, big: bool) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .child(
                div()
                    .text_size(self.z(if big { 24.0 } else { UI_SIZE }))
                    .text_color(color(pal().untyped))
                    .child(label.to_string()),
            )
            .child(
                div()
                    .text_size(self.z(if big { 64.0 } else { 24.0 }))
                    .line_height(self.z(if big { 72.0 } else { 32.0 }))
                    .text_color(color(pal().accent))
                    .child(value),
            )
    }

    fn render_results(&self, done: &Finished, shown: &[Instant], width: f32, window: &mut Window) -> impl IntoElement {
        let s = &done.score;
        let mode = match done.mode {
            Mode::Words(n) => format!("words {n}"),
            Mode::Time(n) => format!("time {n}"),
        };
        let key = |c: char| if c == ' ' { '_' } else { c };
        let slowest =
            done.slowest.iter().map(|(a, b, ms)| format!("{}→{} {ms:.0}ms", key(*a), key(*b))).collect::<Vec<_>>();

        // The hill's lines type out one after another; a line added later starts when it arrives.
        let now = Instant::now();
        let pause = Duration::from_secs_f32((LINE_PAUSE_CHARS as f32 * MS_PER_CHAR) / 1000.0);
        let mut free_at = shown.first().copied().unwrap_or(now);
        let mut lines = Vec::new();
        for (line, added) in done.lines.iter().zip(shown) {
            let start = free_at.max(*added);
            let n = if now > start { revealed(start, now).min(line.len()) } else { 0 };
            if n < line.len() {
                window.request_animation_frame();
            }
            lines.push(div().child(line.chars().take(n).collect::<String>()));
            free_at = start + Duration::from_secs_f32(line.len() as f32 * MS_PER_CHAR / 1000.0) + pause;
        }

        let row = |label: &str, items: Vec<String>, hex: u32| {
            div()
                .flex()
                .gap(self.z(12.0))
                .child(div().w(self.z(130.0)).text_color(color(pal().untyped)).child(label.to_string()))
                .children(items.into_iter().map(move |t| div().text_color(color(hex)).child(t)))
        };
        let mut insights = div().flex().flex_col().gap(self.z(6.0)).text_size(self.z(UI_SIZE));
        if !done.struggling.is_empty() {
            insights = insights.child(row("struggling with", done.struggling.clone(), pal().error));
        }
        if !done.improved.is_empty() {
            insights = insights.child(row("got past", done.improved.clone(), pal().accent));
        }
        if !slowest.is_empty() {
            insights = insights.child(row("slowest", slowest, pal().correct));
        }
        if let Some((word, ms)) = &done.graph.lost {
            insights = insights.child(row("lost most on", vec![format!("{word} +{ms:.0}ms")], pal().correct));
        }

        let headline = div()
            .flex()
            .gap(self.z(48.0))
            .child(self.stat("wpm", format!("{:.0}", s.wpm), true))
            .child(self.stat("acc", format!("{:.0}%", s.accuracy * 100.0), true));
        let headline_w = 300.0 * self.zoom;
        let mut numbers =
            div().flex().gap(self.z(40.0)).child(self.stat("raw", format!("{:.0}", s.raw), false)).child(self.stat(
                "predicted",
                if done.predicted > 0.0 { format!("{:.0}", done.predicted) } else { "-".into() },
                false,
            ));
        if let Some((_, wpm)) = &done.ghost {
            numbers = numbers.child(self.stat("past you", format!("{wpm:.0}"), false));
        }
        div()
            .flex()
            .flex_col()
            .gap(self.z(24.0))
            .child(
                div()
                    .flex()
                    .items_start()
                    .gap(self.z(24.0))
                    .child(div().w(px(headline_w)).flex_none().child(headline))
                    .child(self.render_run_graph(done, (width - headline_w - 24.0 * self.zoom).max(200.0))),
            )
            .child(numbers.child(self.stat("errors", s.errors.to_string(), false)).child(self.stat(
                "mode",
                if done.boss { "boss".into() } else { mode },
                false,
            )))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(self.z(6.0))
                    .min_h(self.z(VOICE_SIZE * 1.4 * 3.0))
                    .text_size(self.z(VOICE_SIZE))
                    .text_color(color(pal().correct))
                    .children(lines),
            )
            .child(insights)
    }

    fn render_stats(&self, width: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let Screen::Stats(st) = &self.screen else { unreachable!() };
        div()
            .flex()
            .flex_col()
            .gap(self.z(28.0))
            .child(self.render_lifetime(&st.lifetime))
            .child(self.render_chart(st, width, cx))
            .child(
                div()
                    .flex()
                    .justify_between()
                    .gap(self.z(32.0))
                    .child(self.render_selected(st))
                    .child(self.render_keyboard(&st.heat)),
            )
            .child(
                div()
                    .text_size(self.z(12.0))
                    .text_color(color(pal().untyped))
                    .child("← → choose   ·   enter play against it   ·   esc back"),
            )
    }

    fn render_lifetime(&self, life: &Lifetime) -> impl IntoElement {
        let trend = match (life.last10, life.previous10) {
            (Some(now), Some(before)) if (now - before).abs() >= 0.5 => {
                format!(" {}{:.0}", if now > before { "↑" } else { "↓" }, (now - before).abs())
            }
            _ => String::new(),
        };
        let or_dash = |v: Option<String>| v.unwrap_or_else(|| "-".into());
        let stats = [
            ("runs", life.runs.to_string()),
            ("time typing", duration(life.seconds)),
            ("keystrokes", thousands(life.keystrokes)),
            ("best", or_dash(life.best.as_ref().map(|(wpm, mode)| format!("{wpm:.0} {mode}")))),
            ("last 10 avg", or_dash(life.last10.map(|w| format!("{w:.0}{trend}")))),
            ("accuracy", or_dash(life.accuracy10.map(|a| format!("{:.1}%", a * 100.0)))),
            ("vs the hill", format!("{}-{}", life.wins, life.losses)),
            ("best streak", life.longest_streak.to_string()),
            ("bosses", format!("{}-{}", life.boss_wins, life.boss_losses)),
        ];
        // Three rows of three equal columns.
        let row = |cells: &[(&str, String)]| {
            div().flex().children(
                cells.iter().map(|(label, value)| div().flex_1().child(self.stat(label, value.clone(), false))),
            )
        };
        div().flex().flex_col().gap(self.z(16.0)).w_full().children(stats.chunks(3).map(row))
    }

    fn render_chart(&self, st: &Stats, width: f32, cx: &mut Context<Self>) -> impl IntoElement {
        let mut values: Vec<f64> = st.checkpoints.iter().map(|c| c.reference_wpm).collect();
        values.extend(st.live_wpm);
        let caption = div()
            .text_size(self.z(12.0))
            .text_color(color(pal().untyped))
            .child("how fast the hill thinks you are on everyday words, checkpoint by checkpoint");
        if st.checkpoints.is_empty() {
            let runs = self.session.store.run_count().unwrap_or(0);
            let left = CHECKPOINT_EVERY - runs % CHECKPOINT_EVERY;
            return div()
                .flex()
                .flex_col()
                .gap(self.z(8.0))
                .child(caption)
                .child(
                    div()
                        .text_size(self.z(18.0))
                        .text_color(color(pal().correct))
                        .child(format!("{left} runs until the first checkpoint")),
                )
                .into_any_element();
        }

        let height = CHART_HEIGHT * self.zoom;
        let (lo, hi) = values.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let pad = ((hi - lo) * 0.15).max(5.0);
        let (lo, hi) = (lo - pad, hi + pad);
        let n = values.len();
        let (left, right, top, bottom) = (44.0 * self.zoom, 16.0 * self.zoom, 12.0 * self.zoom, 28.0 * self.zoom);
        let spots: Vec<Point<Pixels>> = (0..n)
            .map(|i| {
                let x = left + if n == 1 { 0.0 } else { i as f32 * (width - left - right) / (n - 1) as f32 };
                let y = top + (1.0 - ((values[i] - lo) / (hi - lo)) as f32) * (height - top - bottom);
                point(px(x), px(y))
            })
            .collect();
        let line = spots.clone();
        let stroke = self.z(2.0);
        let r = 6.0 * self.zoom;
        let small = self.z(11.0);

        let dots = spots.iter().enumerate().map(|(i, p)| {
            div()
                .id(("dot", i))
                .absolute()
                .left(p.x - px(r))
                .top(p.y - px(r))
                .size(px(2.0 * r))
                .rounded_full()
                .cursor_pointer()
                .bg(color(if i == st.selected { pal().accent } else { pal().untyped }))
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| this.select(i, cx)))
        });
        // Run numbers under the first, last and selected dots only, so they never crowd.
        let labels =
            spots.iter().enumerate().filter(|(i, _)| *i == 0 || *i == n - 1 || *i == st.selected).map(|(i, p)| {
                let label = st.checkpoints.get(i).map_or("now".to_string(), |c| format!("run {}", c.run_count));
                div()
                    .absolute()
                    .left(p.x - px(30.0 * self.zoom))
                    .w(px(60.0 * self.zoom))
                    .top(px(height - bottom + 8.0 * self.zoom))
                    .flex()
                    .justify_center()
                    .text_size(small)
                    .text_color(color(if i == st.selected { pal().accent } else { pal().untyped }))
                    .child(label)
            });
        let axis = |v: f64, y: f32| {
            div()
                .absolute()
                .left(px(0.0))
                .top(px(y - 7.0 * self.zoom))
                .text_size(small)
                .text_color(color(pal().untyped))
                .child(format!("{v:.0}"))
        };

        div()
            .flex()
            .flex_col()
            .gap(self.z(8.0))
            .child(caption)
            .child(
                div()
                    .relative()
                    .w(px(width))
                    .h(px(height))
                    .child(
                        canvas(
                            |_, _, _| {},
                            move |bounds, _, window, _| {
                                let mut path = PathBuilder::stroke(stroke);
                                for (i, p) in line.iter().enumerate() {
                                    let p = bounds.origin + *p;
                                    if i == 0 { path.move_to(p) } else { path.line_to(p) }
                                }
                                if let Ok(path) = path.build() {
                                    window.paint_path(path, rgb(pal().untyped));
                                }
                            },
                        )
                        .size_full(),
                    )
                    .child(axis(hi, top))
                    .child(axis(lo, height - bottom))
                    .children(dots)
                    .children(labels),
            )
            .into_any_element()
    }

    /// Cumulative wpm through the run: you (solid), the hill's committed pace (dashed), the
    /// ghost (dim), with your mistakes marked along the bottom.
    fn render_run_graph(&self, done: &Finished, width: f32) -> impl IntoElement {
        let g = &done.graph;
        let height = 150.0 * self.zoom;
        let (left, bottom, top) = (32.0 * self.zoom, 20.0 * self.zoom, 8.0 * self.zoom);
        let all = g.you.iter().chain(&g.hill).chain(&g.ghost).map(|p| p.1);
        let (lo, hi) = all.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        if g.you.is_empty() || !lo.is_finite() {
            return div().into_any_element();
        }
        let pad = ((hi - lo) * 0.1).max(5.0);
        let (lo, hi) = ((lo - pad).max(0.0), hi + pad);
        let to_xy = move |(t, v): (f32, f32)| {
            point(
                px(left + t / g.seconds.max(0.1) * (width - left)),
                px(top + (1.0 - (v - lo) / (hi - lo)) * (height - top - bottom)),
            )
        };
        let pts = |line: &[(f32, f32)]| line.iter().map(|&p| to_xy(p)).collect::<Vec<_>>();
        let (you, hill, ghost) = (pts(&g.you), pts(&g.hill), pts(&g.ghost));
        let (stroke, thin) = (px(2.0 * self.zoom), px(1.5 * self.zoom));
        let dash = (6.0 * self.zoom, 4.0 * self.zoom);
        let (you_c, hill_c, ghost_c) = (pal().accent, pal().untyped, mix(pal().bg, pal().untyped, 0.55));
        let small = self.z(11.0);
        let label = |text: String, x: f32, y: f32| {
            div().absolute().left(px(x)).top(px(y)).text_size(small).text_color(color(pal().untyped)).child(text)
        };
        let dot = 4.0 * self.zoom;
        let errors = g.errors.iter().map(|&t| {
            let x = f32::from(to_xy((t, lo)).x);
            div()
                .absolute()
                .left(px(x - dot / 2.0))
                .top(px(height - bottom - dot / 2.0))
                .size(px(dot))
                .rounded_full()
                .bg(color(pal().error))
        });
        let key = |name: String, c: Hsla, dashed: bool| {
            div()
                .flex()
                .items_center()
                .gap(self.z(6.0))
                .child(div().w(self.z(14.0)).h(self.z(2.0)).bg(c).opacity(if dashed { 0.7 } else { 1.0 }))
                .child(name)
        };
        let mut legend = div().flex().gap(self.z(16.0)).text_size(small).text_color(color(pal().untyped)).child(key(
            "you".into(),
            color(you_c),
            false,
        ));
        if !hill.is_empty() {
            legend = legend.child(key("the hill".into(), color(hill_c), true));
        }
        if let Some((label, _)) = &done.ghost {
            legend = legend.child(key(format!("past you, {label}"), ghost_c, false));
        }
        div()
            .flex()
            .flex_col()
            .gap(self.z(6.0))
            .child(
                div()
                    .relative()
                    .w(px(width))
                    .h(px(height))
                    .child(
                        canvas(
                            |_, _, _| {},
                            move |bounds, _, window, _| {
                                let shift =
                                    |ps: &[Point<Pixels>]| ps.iter().map(|&p| bounds.origin + p).collect::<Vec<_>>();
                                if let Some(p) = polyline(&shift(&ghost), thin, None) {
                                    window.paint_path(p, ghost_c);
                                }
                                if let Some(p) = polyline(&shift(&hill), thin, Some(dash)) {
                                    window.paint_path(p, rgb(hill_c));
                                }
                                if let Some(p) = polyline(&shift(&you), stroke, None) {
                                    window.paint_path(p, rgb(you_c));
                                }
                            },
                        )
                        .size_full(),
                    )
                    .child(label(format!("{hi:.0}"), 0.0, top - 7.0 * self.zoom))
                    .child(label(format!("{lo:.0}"), 0.0, height - bottom - 7.0 * self.zoom))
                    .child(label("0s".into(), left, height - bottom + 4.0 * self.zoom))
                    .child(label(
                        format!("{:.0}s", g.seconds),
                        width - 24.0 * self.zoom,
                        height - bottom + 4.0 * self.zoom,
                    ))
                    .children(errors),
            )
            .child(legend)
            .into_any_element()
    }

    fn render_selected(&self, st: &Stats) -> impl IntoElement {
        let line = |text: String, hex: u32| div().text_color(color(hex)).child(text);
        let mut panel = div().flex().flex_col().gap(self.z(8.0)).text_size(self.z(UI_SIZE)).max_w(self.z(420.0));
        match st.checkpoints.get(st.selected) {
            Some(c) => {
                panel = panel
                    .child(div().text_size(self.z(18.0)).text_color(color(pal().accent)).child(format!(
                        "{} · run {}",
                        c.date(),
                        c.run_count
                    )))
                    .child(line(
                        format!("thought you'd do {:.0} wpm on everyday words", c.reference_wpm),
                        pal().correct,
                    ));
                panel = panel.child(match c.best {
                    Some((wpm, predicted)) => line(
                        format!("played {} times · your best {wpm:.0}, it predicted {predicted:.0}", c.played),
                        pal().correct,
                    ),
                    None => line("you haven't played against it yet".into(), pal().untyped),
                });
            }
            None => {
                panel = panel
                    .child(
                        div()
                            .text_size(self.z(18.0))
                            .text_color(color(pal().accent))
                            .child("live · learning from every run"),
                    )
                    .child(line(
                        format!("thinks you'd do {:.0} wpm on everyday words", st.live_wpm.unwrap_or(0.0)),
                        pal().correct,
                    ));
            }
        }
        panel.child(
            div()
                .flex()
                .gap(self.z(10.0))
                .child(div().text_color(color(pal().untyped)).child("weakest at"))
                .children(st.tags.iter().map(|t| div().text_color(color(pal().error)).child(t.clone()))),
        )
    }

    /// QWERTY with each key tinted by how slow the selected model expects it to be.
    fn render_keyboard(&self, heat: &[(char, Option<f32>)]) -> impl IntoElement {
        let known = heat.iter().filter_map(|(_, c)| *c);
        let (lo, hi) = known.fold((f32::MAX, f32::MIN), |(lo, hi), c| (lo.min(c), hi.max(c)));
        let size = 30.0 * self.zoom;
        let gap = 4.0 * self.zoom;
        let mut keys = heat.iter();
        let rows = learn::KEYBOARD.iter().enumerate().map(|(r, row)| {
            let indent = [0.0, 0.25, 0.75][r] * (size + gap);
            div().flex().gap(px(gap)).pl(px(indent)).children(row.chars().map(|_| {
                let (key, cost) = keys.next().copied().unwrap_or((' ', None));
                // Unseen keys stay hollow rather than showing a made-up estimate.
                let t = cost.map_or(0.0, |c| if hi > lo { (c - lo) / (hi - lo) } else { 0.0 });
                div()
                    .size(px(size))
                    .rounded(px(4.0 * self.zoom))
                    .when(cost.is_some(), |d| d.bg(mix(pal().key_bg, pal().error, t)))
                    .when(cost.is_none(), |d| d.border_1().border_color(color(pal().key_bg)))
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_size(self.z(UI_SIZE))
                    .text_color(color(if t > 0.6 { pal().correct } else { pal().untyped }))
                    .child(key.to_string())
            }))
        });
        div().flex().flex_col().gap(px(gap)).children(rows).child(
            div()
                .pt(self.z(4.0))
                .text_size(self.z(11.0))
                .text_color(color(pal().untyped))
                .child("redder = slower, in its eyes · outlined = not enough data"),
        )
    }
}

/// A stroked polyline; with `dash`, drawn as (on, off) dashes along its length.
fn polyline(points: &[Point<Pixels>], width: Pixels, dash: Option<(f32, f32)>) -> Option<gpui::Path<Pixels>> {
    let mut path = PathBuilder::stroke(width);
    let &first = points.first()?;
    match dash {
        None => {
            path.move_to(first);
            points[1..].iter().for_each(|&p| path.line_to(p));
        }
        Some((on, off)) => {
            // Walk the line, alternating pen down and pen up, carrying the phase across corners.
            let (mut phase, mut down) = (on, true);
            for pair in points.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let (dx, dy) = (f32::from(b.x - a.x), f32::from(b.y - a.y));
                let len = (dx * dx + dy * dy).sqrt();
                let mut walked = 0.0;
                while walked < len {
                    let step = phase.min(len - walked);
                    let at = |d: f32| point(a.x + px(dx * d / len), a.y + px(dy * d / len));
                    if down {
                        path.move_to(at(walked));
                        path.line_to(at(walked + step));
                    }
                    walked += step;
                    phase -= step;
                    if phase <= 0.0 {
                        down = !down;
                        phase = if down { on } else { off };
                    }
                }
            }
        }
    }
    path.build().ok()
}

fn mix(a: u32, b: u32, t: f32) -> Hsla {
    let ch = |c: u32, shift: u32| ((c >> shift) & 0xff) as f32;
    let lerp = |shift| ch(a, shift) + (ch(b, shift) - ch(a, shift)) * t.clamp(0.0, 1.0);
    rgb(((lerp(16) as u32) << 16) | ((lerp(8) as u32) << 8) | lerp(0) as u32).into()
}

fn duration(seconds: f64) -> String {
    let s = seconds.round() as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {}s", s / 60, s % 60),
        _ => format!("{}h {}m", s / 3600, s % 3600 / 60),
    }
}

fn thousands(n: i64) -> String {
    if n >= 10_000 { format!("{:.1}k", n as f64 / 1000.0) } else { n.to_string() }
}

impl Render for TheHill {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let viewport = window.viewport_size();
        let width = (f32::from(viewport.width) - 96.0).min(MAX_WIDTH * self.zoom);
        let mono: Font = font(self.font.clone());
        let font_id = window.text_system().resolve_font(&mono);
        let text_size = TEXT_SIZE * self.zoom;
        let char_w =
            window.text_system().advance(font_id, px(text_size), 'm').map_or(text_size * 0.6, |s| f32::from(s.width));

        let body = match &self.screen {
            Screen::Typing => div()
                .flex()
                .flex_col()
                .gap(self.z(8.0))
                .child(
                    div().h(self.z(32.0)).text_size(self.z(24.0)).text_color(color(pal().accent)).child(self.counter()),
                )
                .child(self.render_typing(width, char_w, window))
                .child(self.render_speech(window))
                .into_any_element(),
            Screen::Results { done, shown } => self.render_results(done, shown, width, window).into_any_element(),
            Screen::Stats(_) => self.render_stats(width, cx).into_any_element(),
        };
        let hint = match self.screen {
            Screen::Typing => {
                "tab new text  ·  esc restart  ·  ctrl+s stats  ·  ctrl+h hill  ·  ctrl+g ghost  ·  ctrl+t theme  ·  ctrl +/- zoom"
            }
            Screen::Results { .. } => "enter new text  ·  esc retry  ·  s stats",
            Screen::Stats(_) => "",
        };
        let learning = if self.training { "  ·  learning" } else { "" };

        div()
            .track_focus(&self.focus)
            .on_key_down(cx.listener(Self::on_key))
            .size_full()
            .bg(color(pal().bg))
            .font_family(self.font.clone())
            .flex()
            .flex_col()
            .items_center()
            .justify_between()
            .py(self.z(32.0))
            .pt(self.z(32.0) + px(MAC_TITLEBAR))
            // The strip the titlebar would occupy: double-click zooms like any mac window.
            .when(MAC_TITLEBAR > 0.0, |d| {
                d.child(div().id("titlebar").absolute().top_0().left_0().w_full().h(px(MAC_TITLEBAR)).on_click(
                    |e: &ClickEvent, window, _| {
                        if e.click_count() == 2 {
                            window.titlebar_double_click();
                        }
                    },
                ))
            })
            .child(self.render_bar(cx))
            // Centred when it fits, scrollable when zoomed past the window height.
            .child(
                div()
                    .id("body")
                    .flex_1()
                    .min_h(px(0.0))
                    .w_full()
                    .my(self.z(24.0))
                    .overflow_y_scroll()
                    .flex()
                    .flex_col()
                    .items_center()
                    .child(div().my_auto().w(px(width)).flex().justify_center().child(body)),
            )
            .child(div().text_size(self.z(12.0)).text_color(color(pal().untyped)).child(format!("{hint}{learning}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caret_glides_without_overshoot_and_settles() {
        let start = Instant::now();
        let mut caret = Caret::new(start);
        let frame = Duration::from_millis(8);
        let mut now = start;
        let mut last = 0.0;
        // Keys every 80 ms (150 wpm), 17 px apart: the caret keeps moving forward, never back.
        for key in 1..=10 {
            caret.retarget(point(px(key as f32 * 17.0), px(0.0)), now);
            for _ in 0..10 {
                now += frame;
                let (pos, _) = caret.step(now);
                let x = f32::from(pos.x);
                assert!(x >= last - 1e-3 && x <= key as f32 * 17.0 + 1e-3, "x {x} after {last}");
                last = x;
            }
        }
        for _ in 0..40 {
            now += frame;
            caret.step(now);
        }
        assert_eq!(caret.step(now + frame), (point(px(170.0), px(0.0)), false));
        // A new line jumps.
        caret.retarget(point(px(0.0), px(40.0)), now);
        assert_eq!(caret.step(now + frame).0, point(px(0.0), px(40.0)));
    }
}
