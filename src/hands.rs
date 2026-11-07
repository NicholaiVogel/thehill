//! Hands and fingers: which finger types each key on a standard QWERTY touch-typing layout,
//! and how each finger, hand and motion is doing in the keystroke log.
//!
//! Uses the same rules as training: only the first attempt at each text position counts,
//! and an interval is timed only after a clean keypress and when it's under 1.5 s.

use crate::model::RunLog;
use crate::typing::{Event, Key};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Finger {
    LPinky,
    LRing,
    LMiddle,
    LIndex,
    RIndex,
    RMiddle,
    RRing,
    RPinky,
}

pub const FINGERS: [Finger; 8] = [
    Finger::LPinky,
    Finger::LRing,
    Finger::LMiddle,
    Finger::LIndex,
    Finger::RIndex,
    Finger::RMiddle,
    Finger::RRing,
    Finger::RPinky,
];

/// Keys per finger, unshifted.
const KEYS: [&str; 8] = ["`1qaz", "2wsx", "3edc", "45rtfgvb", "67yuhjnm", "8ik,", "9ol.", "0p;/-=[]\\'"];
const SHIFTED: &str = "~!@#$%^&*()_+{}|:\"<>?";
const UNSHIFTED: &str = "`1234567890-=[]\\;',./";
/// Keys at the same spot on the other hand.
const MIRROR: [(&str, &str); 3] = [("qwert", "poiuy"), ("asdfg", ";lkjh"), ("zxcvb", "/.,mn")];
/// A finger's median needs at least this many timed keystrokes to be shown.
pub const MIN_SAMPLES: usize = 20;

impl Finger {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn left(self) -> bool {
        self.index() < 4
    }

    pub fn name(self) -> &'static str {
        [
            "left pinky",
            "left ring",
            "left middle",
            "left index",
            "right index",
            "right middle",
            "right ring",
            "right pinky",
        ][self.index()]
    }

    pub fn short(self) -> &'static str {
        ["pinky", "ring", "middle", "index", "index", "middle", "ring", "pinky"][self.index()]
    }
}

fn base(c: char) -> char {
    let c = c.to_ascii_lowercase();
    SHIFTED.find(c).and_then(|i| UNSHIFTED.chars().nth(i)).unwrap_or(c)
}

/// The finger that types `c`; `None` for space and anything off the main keys.
pub fn finger(c: char) -> Option<Finger> {
    let c = base(c);
    KEYS.iter().position(|keys| keys.contains(c)).map(|i| FINGERS[i])
}

fn mirror(c: char) -> Option<char> {
    let c = base(c);
    MIRROR.iter().find_map(|(l, r)| {
        l.find(c).and_then(|i| r.chars().nth(i)).or_else(|| r.find(c).and_then(|i| l.chars().nth(i)))
    })
}

/// What kind of mistake a wrong key was.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slip {
    /// The right key with or without shift when it should have been the other way.
    Shift,
    /// Typed the next letter before this one.
    Rushed,
    /// The finger beside the right one, same hand.
    Neighbour,
    /// The right finger, wrong key (usually the wrong row).
    WrongRow,
    /// A finger further over on the same hand.
    SameHand,
    /// The same spot on the other hand.
    Mirror,
    /// Some other key on the other hand.
    OtherHand,
    Other,
}

pub const SLIPS: [Slip; 8] = [
    Slip::Shift,
    Slip::Rushed,
    Slip::Neighbour,
    Slip::WrongRow,
    Slip::SameHand,
    Slip::Mirror,
    Slip::OtherHand,
    Slip::Other,
];

impl Slip {
    pub fn name(self) -> &'static str {
        match self {
            Slip::Shift => "shift",
            Slip::Rushed => "rushed ahead",
            Slip::Neighbour => "finger next door",
            Slip::WrongRow => "right finger, wrong key",
            Slip::SameHand => "same hand, further over",
            Slip::Mirror => "mirror key, other hand",
            Slip::OtherHand => "other hand",
            Slip::Other => "other",
        }
    }
}

fn classify(expected: char, typed: char, next: Option<char>) -> Slip {
    let (e, t) = (base(expected), base(typed));
    if e == t {
        return Slip::Shift;
    }
    if next.is_some_and(|n| base(n) == t) {
        return Slip::Rushed;
    }
    match (finger(e), finger(t)) {
        (Some(a), Some(b)) if a == b => Slip::WrongRow,
        _ if mirror(e) == Some(t) => Slip::Mirror,
        (Some(a), Some(b)) if a.left() == b.left() && a.index().abs_diff(b.index()) == 1 => Slip::Neighbour,
        (Some(a), Some(b)) if a.left() == b.left() => Slip::SameHand,
        (Some(a), Some(b)) if a.left() != b.left() => Slip::OtherHand,
        _ => Slip::Other,
    }
}

/// Raw counts and timings, added up run by run.
#[derive(Default)]
pub struct Tally {
    ms: [Vec<f32>; 8],
    attempts: [u32; 8],
    errors: [u32; 8],
    slips: [u32; 8],
    /// Time (ms) spent reaching each hand's letters: every interval before a first attempt,
    /// clean or not, except pauses.
    reach: [f64; 2],
    confusions: HashMap<(char, char), u32>,
    /// Intervals between two letters on the same finger, and on different fingers.
    same_finger: Vec<f32>,
    other_finger: Vec<f32>,
    /// From a right-hand letter into a left-hand one, and the reverse.
    into_left: Vec<f32>,
    into_right: Vec<f32>,
    /// Same hand, different finger: towards the index finger, and away from it.
    inward: Vec<f32>,
    outward: Vec<f32>,
}

impl Tally {
    pub fn add(&mut self, text: &[char], events: &[Event]) {
        let mut attempted = std::collections::HashSet::new();
        let mut prev: Option<(f64, bool, usize)> = None;
        for e in events {
            let is_char = matches!(e.key, Key::Char(_));
            if let (Key::Char(typed), Some(expected)) = (e.key, e.expected)
                && attempted.insert(e.idx)
                && let Some(f) = finger(expected)
            {
                self.attempts[f.index()] += 1;
                if !e.correct {
                    self.errors[f.index()] += 1;
                    let slip = classify(expected, typed, text.get(e.idx + 1).copied());
                    self.slips[SLIPS.iter().position(|s| *s == slip).unwrap()] += 1;
                    if typed != ' ' && slip != Slip::Shift {
                        *self.confusions.entry((base(expected), base(typed))).or_default() += 1;
                    }
                }
                if let Some((t, _, _)) = prev
                    && e.t - t > 0.0
                    && e.t - t <= 1500.0
                {
                    self.reach[usize::from(!f.left())] += e.t - t;
                }
                let timed = match prev {
                    Some((t, true, _)) if e.correct && e.t - t > 0.0 && e.t - t <= 1500.0 => Some((e.t - t) as f32),
                    _ => None,
                };
                if let Some(ms) = timed {
                    self.ms[f.index()].push(ms);
                    let before = e.idx.checked_sub(1).and_then(|i| text.get(i)).copied();
                    if let Some(p) = before.and_then(finger).filter(|_| prev.is_some_and(|p| p.2 + 1 == e.idx)) {
                        let same_key = before.map(base) == Some(base(expected));
                        if p == f && !same_key {
                            self.same_finger.push(ms);
                        } else if p != f {
                            self.other_finger.push(ms);
                        }
                        match (p.left(), f.left()) {
                            (false, true) => self.into_left.push(ms),
                            (true, false) => self.into_right.push(ms),
                            _ if p != f => {
                                // Towards the index finger is inward on both hands.
                                let towards = if f.left() { f.index() > p.index() } else { f.index() < p.index() };
                                if towards { self.inward.push(ms) } else { self.outward.push(ms) }
                            }
                            _ => {}
                        }
                    }
                }
            }
            prev = Some((e.t, is_char && e.correct, e.idx));
        }
    }
}

fn median(xs: &[f32]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(f32::total_cmp);
    let m = v.len() / 2;
    Some(if v.len() % 2 == 1 { v[m] as f64 } else { (v[m - 1] as f64 + v[m] as f64) / 2.0 })
}

/// `median` when there are at least `min` samples.
fn median_of(xs: &[f32], min: usize) -> Option<f64> {
    if xs.len() >= min { median(xs) } else { None }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct FingerStat {
    /// Median ms per keystroke; `None` without enough samples.
    pub ms: Option<f64>,
    pub error_rate: Option<f64>,
    pub keys: u32,
    pub errors: u32,
}

/// A summary of everything in a tally.
#[derive(Clone, Debug, Default)]
pub struct Hands {
    pub fingers: [FingerStat; 8],
    /// Left, right: median ms per keystroke and error rate.
    pub hands: [FingerStat; 2],
    /// Share of letters each hand typed, and of the timed time it spent.
    pub work: [f64; 2],
    pub time: [f64; 2],
    /// Each kind of mistake with its share of all mistakes, most common first.
    pub slips: Vec<(Slip, f64)>,
    pub mistakes: u32,
    /// Most common (wanted, typed) mix-ups.
    pub confusions: Vec<(char, char, u32)>,
    /// Median ms: same finger twice vs a different finger, with how many same-finger pairs.
    pub same_finger: Option<(f64, f64, usize)>,
    /// Median ms into the left hand from the right, and into the right from the left.
    pub switch: Option<(f64, f64)>,
    /// Median ms rolling inward (towards the index finger) and outward.
    pub rolls: Option<(f64, f64)>,
}

impl Hands {
    /// The slowest finger with enough data.
    pub fn slowest(&self) -> Option<(Finger, f64)> {
        FINGERS.iter().filter_map(|&f| Some((f, self.fingers[f.index()].ms?))).max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The finger making the largest share of the mistakes.
    pub fn sloppiest(&self) -> Option<(Finger, f64)> {
        let total = self.mistakes as f64;
        FINGERS
            .iter()
            .map(|&f| (f, self.fingers[f.index()].errors as f64 / total))
            .filter(|(_, share)| *share > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

impl Tally {
    /// Summarizes, showing a median only from `min` timed keystrokes up.
    pub fn summary(&self, min: usize) -> Hands {
        let mut h = Hands::default();
        let (mut hand_ms, mut hand_keys, mut hand_errors): ([Vec<f32>; 2], [u32; 2], [u32; 2]) = Default::default();
        let hand_time = self.reach;
        for f in FINGERS {
            let i = f.index();
            let side = usize::from(!f.left());
            h.fingers[i] = FingerStat {
                ms: median_of(&self.ms[i], min),
                error_rate: (self.attempts[i] as usize >= min).then(|| self.errors[i] as f64 / self.attempts[i] as f64),
                keys: self.attempts[i],
                errors: self.errors[i],
            };
            hand_ms[side].extend(&self.ms[i]);
            hand_keys[side] += self.attempts[i];
            hand_errors[side] += self.errors[i];
        }
        for side in 0..2 {
            h.hands[side] = FingerStat {
                ms: median_of(&hand_ms[side], min),
                error_rate: (hand_keys[side] > 0).then(|| hand_errors[side] as f64 / hand_keys[side] as f64),
                keys: hand_keys[side],
                errors: hand_errors[side],
            };
        }
        let keys = (hand_keys[0] + hand_keys[1]).max(1) as f64;
        let time = (hand_time[0] + hand_time[1]).max(1.0);
        h.work = [hand_keys[0] as f64 / keys, hand_keys[1] as f64 / keys];
        h.time = [hand_time[0] / time, hand_time[1] / time];
        h.mistakes = self.slips.iter().sum();
        let mut slips: Vec<(Slip, f64)> = SLIPS
            .iter()
            .zip(self.slips)
            .filter(|(_, n)| *n > 0)
            .map(|(s, n)| (*s, n as f64 / h.mistakes.max(1) as f64))
            .collect();
        slips.sort_by(|a, b| b.1.total_cmp(&a.1));
        h.slips = slips;
        let mut confusions: Vec<(char, char, u32)> = self.confusions.iter().map(|(&(a, b), &n)| (a, b, n)).collect();
        confusions.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        confusions.truncate(5);
        h.confusions = confusions;
        let pair = |a: &[f32], b: &[f32]| Some((median_of(a, min)?, median_of(b, min)?));
        h.same_finger = pair(&self.same_finger, &self.other_finger).map(|(s, o)| (s, o, self.same_finger.len()));
        h.switch = pair(&self.into_left, &self.into_right);
        h.rolls = pair(&self.inward, &self.outward);
        h
    }
}

pub fn of_runs(runs: &[RunLog]) -> Hands {
    let mut t = Tally::default();
    runs.iter().for_each(|r| t.add(&r.text, &r.events));
    t.summary(MIN_SAMPLES)
}

/// What one run says about your hands, for the results screen and the hill.
#[derive(Clone, Debug)]
pub struct RunHands {
    pub left_ms: f64,
    pub right_ms: f64,
    /// Share of the letters the left hand typed, and of the time it took.
    pub left_work: f64,
    pub left_time: f64,
    pub slowest: Option<(Finger, f64)>,
    /// The mix-up made most often in this run, if at least twice.
    pub confusion: Option<(char, char, u32)>,
    pub rushed: u32,
    /// Same-finger pairs typed, and how much longer each took than other pairs (ms).
    pub same_finger: Option<(usize, f64)>,
}

/// Timed keystrokes a finger needs in one run to be called the slowest.
const RUN_MIN: usize = 4;

pub fn of_run(text: &[char], events: &[Event]) -> Option<RunHands> {
    let mut t = Tally::default();
    t.add(text, events);
    let h = t.summary(RUN_MIN);
    Some(RunHands {
        left_ms: h.hands[0].ms?,
        right_ms: h.hands[1].ms?,
        left_work: h.work[0],
        left_time: h.time[0],
        slowest: h.slowest(),
        confusion: h.confusions.first().copied().filter(|c| c.2 >= 2),
        rushed: t.slips[0],
        same_finger: median(&t.same_finger).zip(median(&t.other_finger)).map(|(s, o)| (t.same_finger.len(), s - o)),
    })
}

/// How many same-finger pairs (different keys) a word has.
pub fn same_finger_pairs(word: &str) -> usize {
    let chars: Vec<char> = word.chars().collect();
    chars
        .windows(2)
        .filter(|p| base(p[0]) != base(p[1]) && finger(p[0]).is_some() && finger(p[0]) == finger(p[1]))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(text: &str, keys: &[(char, f64)]) -> Vec<Event> {
        let chars: Vec<char> = text.chars().collect();
        keys.iter()
            .enumerate()
            .map(|(idx, &(c, t))| {
                let expected = chars.get(idx).copied();
                Event { t, idx, expected, key: Key::Char(c), correct: Some(c) == expected }
            })
            .collect()
    }

    #[test]
    fn fingers_follow_touch_typing() {
        assert_eq!(finger('q'), Some(Finger::LPinky));
        assert_eq!(finger('B'), Some(Finger::LIndex));
        assert_eq!(finger('y'), Some(Finger::RIndex));
        assert_eq!(finger('?'), Some(Finger::RPinky), "shifted keys use their key's finger");
        assert_eq!(finger(' '), None);
        assert_eq!(mirror('e'), Some('i'));
        assert_eq!(mirror('m'), Some('v'));
        assert_eq!(same_finger_pairs("decide"), 3, "de, ce, ed: middle finger twice");
        assert_eq!(same_finger_pairs("look"), 1, "lo; oo is the same key");
    }

    #[test]
    fn mistakes_are_sorted_into_kinds() {
        assert_eq!(classify('t', 'h', Some('h')), Slip::Rushed);
        assert_eq!(classify('u', 'i', Some('x')), Slip::Neighbour);
        assert_eq!(classify('u', 'y', None), Slip::WrongRow);
        assert_eq!(classify('e', 'i', None), Slip::Mirror);
        assert_eq!(classify('a', 'k', None), Slip::OtherHand);
        assert_eq!(classify('a', 'f', None), Slip::SameHand);
        assert_eq!(classify('A', 'a', Some('a')), Slip::Shift, "the right key, just without shift");
    }

    #[test]
    fn shift_slips_and_time_on_fumbled_keys_count() {
        // Capitals typed lowercase: the right key, so no mix-up and no "rushed".
        let text: Vec<char> = "Aa Aa".chars().collect();
        let keys: Vec<(char, f64)> = "aa aa".chars().enumerate().map(|(i, c)| (c, i as f64 * 100.0)).collect();
        let mut tally = Tally::default();
        tally.add(&text, &typed("Aa Aa", &keys));
        let h = tally.summary(1);
        assert_eq!(h.slips, vec![(Slip::Shift, 1.0)]);
        assert!(h.confusions.is_empty());
        assert_eq!(h.sloppiest(), Some((Finger::LPinky, 1.0)));
        // A slow, wrong key still took that hand's time.
        let mut tally = Tally::default();
        tally.add(&['a', 'j'], &typed("aj", &[('a', 0.0), ('k', 900.0)]));
        assert_eq!(tally.summary(1).time, [0.0, 1.0]);
    }

    #[test]
    fn hands_motions_and_slips_add_up() {
        let text = "deck fun";
        // Every key 100 ms, except "c" after "e" (same finger) at 300, and a slip: "u" typed "i".
        let mut keys = Vec::new();
        let mut t = 0.0;
        for (i, c) in text.chars().enumerate() {
            t += if i == 2 { 300.0 } else { 100.0 };
            keys.push((if i == 6 { 'i' } else { c }, t));
        }
        let mut tally = Tally::default();
        tally.add(&text.chars().collect::<Vec<_>>(), &typed(text, &keys));
        let h = tally.summary(1);
        assert_eq!(h.mistakes, 1);
        assert_eq!(h.slips, vec![(Slip::Neighbour, 1.0)]);
        assert_eq!(h.confusions, vec![('u', 'i', 1)]);
        assert_eq!(h.fingers[Finger::LMiddle.index()].ms, Some(200.0), "e at 100, c at 300");
        // d-e (100) and e-c (300) are both the left middle finger; c-k is not.
        assert_eq!(h.same_finger, Some((200.0, 100.0, 2)));
        // Left: e, c, f (100 + 300 + 100). Right: k, the wrong u, n (100 each).
        assert_eq!(h.time, [0.625, 0.375]);
        assert_eq!(h.sloppiest(), Some((Finger::RIndex, 1.0)));
        // "n" after the slip isn't timed: recovery, not typing.
        assert_eq!(h.fingers[Finger::RIndex.index()].keys, 2);
        assert_eq!(h.fingers[Finger::RIndex.index()].ms, None, "u was wrong and n came after it");
    }
}
