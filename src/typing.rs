//! The typing test: monkeytype rules, an event log, and the score.

pub const MAX_EXTRA: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Key {
    Char(char),
    Backspace,
    ClearWord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Milliseconds since the first keypress.
    pub t: f64,
    /// Position in the joined text (words separated by single spaces) the keypress aimed at.
    pub idx: usize,
    /// Text char at `idx`; `None` past the end of the text.
    pub expected: Option<char>,
    pub key: Key,
    pub correct: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub wpm: f64,
    pub raw: f64,
    pub accuracy: f64,
    pub errors: usize,
    pub seconds: f64,
}

pub struct Test {
    pub words: Vec<Vec<char>>,
    pub inputs: Vec<Vec<char>>,
    pub current: usize,
    pub events: Vec<Event>,
    pub finished: bool,
    /// The last word was ended with a space, so a correct prefix of it no longer counts.
    submitted: bool,
    offsets: Vec<usize>,
}

impl Test {
    pub fn new(words: Vec<String>) -> Self {
        let mut test = Test {
            words: Vec::new(),
            inputs: Vec::new(),
            current: 0,
            events: Vec::new(),
            finished: false,
            submitted: false,
            offsets: Vec::new(),
        };
        test.extend(words);
        test
    }

    pub fn extend(&mut self, words: Vec<String>) {
        for word in words {
            let offset = match (self.offsets.last(), self.words.last()) {
                (Some(o), Some(w)) => o + w.len() + 1,
                _ => 0,
            };
            self.offsets.push(offset);
            self.words.push(word.chars().collect());
            self.inputs.push(Vec::new());
        }
    }

    pub fn text(&self) -> String {
        self.words.iter().map(|w| w.iter().collect::<String>()).collect::<Vec<_>>().join(" ")
    }

    pub fn started(&self) -> bool {
        !self.events.is_empty()
    }

    fn is_last(&self) -> bool {
        self.current + 1 == self.words.len()
    }

    /// The text position `key` aims at and the char expected there. Past the end of a word,
    /// a space aims at the separator and anything else is an extra with nothing expected.
    fn target(&self, key: Key) -> (usize, Option<char>) {
        let word = &self.words[self.current];
        let typed = self.inputs[self.current].len();
        let idx = self.offsets[self.current] + typed.min(word.len());
        let expected = match word.get(typed) {
            Some(&c) => Some(c),
            None if key == Key::Char(' ') && !self.is_last() => Some(' '),
            None => None,
        };
        (idx, expected)
    }

    fn log(&mut self, t: f64, key: Key) {
        let (idx, expected) = self.target(key);
        let correct = matches!(key, Key::Char(c) if Some(c) == expected);
        self.events.push(Event { t, idx, expected, key, correct });
    }

    pub fn type_char(&mut self, c: char, t: f64) {
        if self.finished {
            return;
        }
        if c == ' ' {
            return self.space(t);
        }
        if self.inputs[self.current].len() >= self.words[self.current].len() + MAX_EXTRA {
            return;
        }
        self.log(t, Key::Char(c));
        self.inputs[self.current].push(c);
        if self.is_last() && self.inputs[self.current] == self.words[self.current] {
            self.finished = true;
        }
    }

    fn space(&mut self, t: f64) {
        if self.inputs[self.current].is_empty() {
            return;
        }
        self.log(t, Key::Char(' '));
        if self.is_last() {
            self.finished = true;
            self.submitted = true;
        } else {
            self.current += 1;
        }
    }

    pub fn backspace(&mut self, t: f64) {
        if self.finished {
            return;
        }
        if !self.inputs[self.current].is_empty() {
            self.log(t, Key::Backspace);
            self.inputs[self.current].pop();
        } else if self.current > 0 && self.inputs[self.current - 1] != self.words[self.current - 1] {
            self.log(t, Key::Backspace);
            self.current -= 1;
        }
    }

    pub fn clear_word(&mut self, t: f64) {
        if self.finished || self.inputs[self.current].is_empty() {
            return;
        }
        self.log(t, Key::ClearWord);
        self.inputs[self.current].clear();
    }

    /// Monkeytype's formulas. `seconds` is the run length (the time limit in time mode).
    pub fn score(&self, seconds: f64) -> Score {
        let minutes = seconds / 60.0;
        let correct_chars = self.correct_chars();
        let presses = self.events.iter().filter(|e| matches!(e.key, Key::Char(_)));
        let (total, correct) = presses.fold((0, 0), |(t, c), e| (t + 1, c + usize::from(e.correct)));
        Score {
            wpm: correct_chars as f64 / 5.0 / minutes,
            raw: total as f64 / 5.0 / minutes,
            accuracy: if total == 0 { 0.0 } else { correct as f64 / total as f64 },
            errors: total - correct,
            seconds,
        }
    }

    /// Chars that count toward wpm: correct words with their spaces, plus a correct prefix
    /// of the word being typed.
    pub fn correct_chars(&self) -> usize {
        let mut correct_chars = 0;
        for i in 0..=self.current {
            let (word, input) = (&self.words[i], &self.inputs[i]);
            let space = usize::from(i < self.current);
            if input == word {
                correct_chars += word.len() + space;
            } else if i == self.current && !self.submitted && word.starts_with(input) {
                correct_chars += input.len();
            }
        }
        correct_chars
    }

    /// Replays the event log on fresh text, giving the counted chars after each event.
    pub fn progress(&self) -> Vec<(f64, usize)> {
        let mut replay = Test::new(self.words.iter().map(|w| w.iter().collect()).collect());
        self.events
            .iter()
            .map(|e| {
                match e.key {
                    Key::Char(c) => replay.type_char(c, e.t),
                    Key::Backspace => replay.backspace(e.t),
                    Key::ClearWord => replay.clear_word(e.t),
                }
                (e.t, replay.correct_chars())
            })
            .collect()
    }

    /// Duration from the first to the last keypress, for words mode.
    pub fn elapsed_ms(&self) -> f64 {
        self.events.last().map_or(0.0, |e| e.t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test(words: &[&str]) -> Test {
        Test::new(words.iter().map(|w| w.to_string()).collect())
    }

    fn type_str(t: &mut Test, s: &str) {
        for c in s.chars() {
            let at = t.events.len() as f64 * 100.0;
            t.type_char(c, at);
        }
    }

    #[test]
    fn perfect_run_finishes_on_last_char() {
        let mut t = test(&["the", "cat"]);
        type_str(&mut t, "the ca");
        assert!(!t.finished);
        type_str(&mut t, "t");
        assert!(t.finished);
        // 7 chars incl. one space, in 12 seconds: 7 / 5 / 0.2 = 7 wpm.
        let s = t.score(12.0);
        assert!((s.wpm - 7.0).abs() < 1e-9);
        assert!((s.raw - 7.0).abs() < 1e-9);
        assert_eq!(s.accuracy, 1.0);
        assert_eq!(s.errors, 0);
    }

    #[test]
    fn errors_count_against_wpm_not_raw() {
        let mut t = test(&["the", "cat", "sat"]);
        type_str(&mut t, "thx cat sat");
        let s = t.score(60.0);
        // "thx" is wrong: correct chars = "cat " + "sat" = 7.
        assert!((s.wpm - 7.0 / 5.0).abs() < 1e-9);
        assert!((s.raw - 11.0 / 5.0).abs() < 1e-9);
        assert!((s.accuracy - 10.0 / 11.0).abs() < 1e-9);
        assert_eq!(s.errors, 1);
    }

    #[test]
    fn space_skips_rest_of_word() {
        let mut t = test(&["hello", "you"]);
        type_str(&mut t, "he ");
        assert_eq!(t.current, 1);
        let e = t.events.last().unwrap();
        assert_eq!((e.idx, e.expected, e.correct), (2, Some('l'), false));
        assert_eq!(t.offsets[t.current], 6);
    }

    #[test]
    fn space_on_empty_word_is_ignored() {
        let mut t = test(&["a", "b"]);
        type_str(&mut t, " ");
        assert!(t.events.is_empty());
        assert_eq!(t.current, 0);
    }

    #[test]
    fn backspace_returns_only_into_wrong_words() {
        let mut t = test(&["ab", "cd", "ef"]);
        type_str(&mut t, "ab ");
        t.backspace(1.0);
        assert_eq!(t.current, 1, "correct word is locked");
        type_str(&mut t, "cx ");
        t.backspace(2.0);
        assert_eq!(t.current, 1, "wrong word can be revisited");
        assert_eq!(t.inputs[1], vec!['c', 'x']);
        t.backspace(3.0);
        type_str(&mut t, "d");
        assert_eq!(t.inputs[1], vec!['c', 'd']);
    }

    #[test]
    fn extras_are_capped_and_logged_at_word_end() {
        let mut t = test(&["ab", "c"]);
        type_str(&mut t, "abxxxxxxxxxxxxxx");
        assert_eq!(t.inputs[0].len(), 2 + MAX_EXTRA);
        let first_extra = &t.events[2];
        assert_eq!((first_extra.idx, first_extra.expected, first_extra.correct), (2, None, false));
    }

    #[test]
    fn space_after_extras_is_a_correct_separator() {
        let mut t = test(&["ab", "cd"]);
        type_str(&mut t, "abx ");
        let space = t.events.last().unwrap();
        assert_eq!((space.idx, space.expected, space.correct), (2, Some(' '), true));
        let s = t.score(60.0);
        assert_eq!(s.errors, 1);
        assert!((s.accuracy - 0.75).abs() < 1e-9);
    }

    #[test]
    fn wrong_last_word_submitted_with_space_scores_nothing() {
        let mut t = test(&["ab", "cd"]);
        type_str(&mut t, "ab c ");
        assert!(t.finished);
        // "ab " counts; the submitted, incomplete "c" does not.
        assert!((t.score(60.0).wpm - 3.0 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn clear_word_empties_input() {
        let mut t = test(&["abc", "d"]);
        type_str(&mut t, "ab");
        t.clear_word(5.0);
        assert!(t.inputs[0].is_empty());
        assert_eq!(t.events.last().unwrap().key, Key::ClearWord);
    }

    #[test]
    fn partial_current_word_counts_in_time_mode() {
        let mut t = test(&["one", "two"]);
        type_str(&mut t, "one tw");
        // The timer ends the run: "one " + "tw" = 6 chars in 6 seconds = 12 wpm.
        t.finished = true;
        assert!((t.score(6.0).wpm - 12.0).abs() < 1e-9);
    }
}
