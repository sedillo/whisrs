//! Deduplication strategies for chunked/sliding-window ASR transcription.
//!
//! Streaming speech-to-text systems process audio in overlapping chunks, which
//! means adjacent or overlapping windows often transcribe the same words
//! multiple times. This crate provides two complementary strategies — each
//! with its own tracker type — to eliminate those duplicates:
//!
//! ## Strategy 1 — Timestamp-based dedup ([`TimestampDedup`])
//!
//! Use this when your ASR provider returns **per-word timestamps** (start/end
//! time for each word). Call [`TimestampDedup::advance_offset`] before each
//! chunk to tell the tracker how much audio has been sent, then pass the words
//! through [`TimestampDedup::filter_words`]. Words whose adjusted start time
//! falls within the already-transcribed range are discarded automatically.
//!
//! Ideal for cloud APIs (e.g. Groq, Deepgram) that return word-level timing.
//!
//! ## Strategy 2 — Text-based anchor dedup ([`TextDedup`])
//!
//! Use this when you have **no timestamps** but receive full-text
//! transcription for each sliding window (e.g. local whisper.cpp). Call
//! [`TextDedup::filter_text`] with each window's complete output. The tracker
//! takes the last N words of the previous window as an "anchor", searches for
//! that anchor anywhere in the new window's text, and returns only the novel
//! suffix.
//!
//! The anchor search is fuzzy (per-word Jaro-Winkler similarity ≥ 0.85) to
//! handle whisper's tendency to slightly rephrase overlapping regions (word
//! insertions, deletions, or substitutions at window boundaries).
//!
//! ## Quick comparison
//!
//! |                 | [`TimestampDedup`]    | [`TextDedup`]      |
//! |-----------------|-----------------------|--------------------|
//! | Input           | Word list + timestamps| Full-text per window |
//! | Requires offset | Yes (`advance_offset`)| No                 |
//! | Fuzzy matching  | No                    | Yes (Jaro-Winkler) |
//! | Best for        | Cloud APIs            | Local whisper      |

/// A word with optional timestamp bounds (seconds from chunk start).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_secs: f64,
    pub end_secs: f64,
}

/// Default tolerance (in seconds) when comparing word start timestamps to
/// the already-transcribed boundary in [`TimestampDedup`].
pub const DEFAULT_OVERLAP_TOLERANCE_SECS: f64 = 0.05;

/// Default maximum number of bytes to keep in [`TextDedup`]'s recent-text
/// buffer between calls.
pub const DEFAULT_MAX_RECENT_CHARS: usize = 500;

/// Tracks transcription progress across multiple chunks using per-word
/// timestamps (Strategy 1).
///
/// Use this when your ASR provider returns word-level start/end times. Words
/// whose adjusted start time falls within the already-transcribed range are
/// dropped.
#[derive(Debug, Clone)]
pub struct TimestampDedup {
    /// The end time (in seconds) of the last word we accepted.
    transcribed_up_to: f64,
    /// Cumulative time offset added to each chunk's timestamps.
    cumulative_offset: f64,
    /// Tolerance (seconds) when comparing word start to `transcribed_up_to`.
    overlap_tolerance_secs: f64,
}

impl TimestampDedup {
    /// Create a new tracker with the default overlap tolerance
    /// ([`DEFAULT_OVERLAP_TOLERANCE_SECS`]).
    pub fn new() -> Self {
        Self {
            transcribed_up_to: 0.0,
            cumulative_offset: 0.0,
            overlap_tolerance_secs: DEFAULT_OVERLAP_TOLERANCE_SECS,
        }
    }

    /// Builder-style setter: configure the overlap tolerance (in seconds).
    ///
    /// A word is accepted if its adjusted `start_secs >= transcribed_up_to -
    /// tolerance`. Larger values are more permissive; smaller values are
    /// stricter. Default: [`DEFAULT_OVERLAP_TOLERANCE_SECS`].
    pub fn with_overlap_tolerance_secs(mut self, tolerance_secs: f64) -> Self {
        self.overlap_tolerance_secs = tolerance_secs;
        self
    }

    /// Add a time offset for the next chunk (the duration of audio already sent).
    pub fn advance_offset(&mut self, chunk_duration_secs: f64) {
        self.cumulative_offset += chunk_duration_secs;
        #[cfg(feature = "logging")]
        log::debug!(
            "dedup: advanced offset by {:.2}s, cumulative = {:.2}s",
            chunk_duration_secs,
            self.cumulative_offset
        );
    }

    /// Filter words from a new chunk, returning only the non-duplicate ones.
    ///
    /// Each word's `start_secs` and `end_secs` are adjusted by the cumulative offset.
    /// Words whose adjusted `start_secs` falls within the already-transcribed
    /// range (minus the configured tolerance) are discarded.
    pub fn filter_words(&mut self, words: &[Word]) -> Vec<Word> {
        let mut accepted = Vec::new();

        for word in words {
            let adjusted_start = word.start_secs + self.cumulative_offset;
            let adjusted_end = word.end_secs + self.cumulative_offset;

            if adjusted_start >= self.transcribed_up_to - self.overlap_tolerance_secs {
                // Accept this word.
                accepted.push(Word {
                    text: word.text.clone(),
                    start_secs: adjusted_start,
                    end_secs: adjusted_end,
                });
                self.transcribed_up_to = adjusted_end;
            }
        }

        #[cfg(feature = "logging")]
        log::debug!(
            "dedup: accepted {}/{} words, transcribed_up_to = {:.2}s",
            accepted.len(),
            words.len(),
            self.transcribed_up_to
        );

        accepted
    }
}

impl Default for TimestampDedup {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks transcription progress across sliding windows using text-anchor
/// matching (Strategy 2).
///
/// Use this when you only get full-text output per window (no timestamps).
/// Call [`TextDedup::filter_text`] with each window's complete transcription
/// and receive only the novel suffix.
#[derive(Debug, Clone)]
pub struct TextDedup {
    /// Previous window's full transcription (for anchor-based text dedup).
    recent_text: String,
    /// Maximum number of bytes to keep in `recent_text` for matching.
    max_recent_chars: usize,
}

impl TextDedup {
    /// Create a new tracker with the default recent-text capacity
    /// ([`DEFAULT_MAX_RECENT_CHARS`]).
    pub fn new() -> Self {
        Self {
            recent_text: String::new(),
            max_recent_chars: DEFAULT_MAX_RECENT_CHARS,
        }
    }

    /// Builder-style setter: configure the maximum byte length of the
    /// recent-text buffer kept between calls.
    ///
    /// Larger values let the anchor match against longer history at the cost
    /// of more memory. Default: [`DEFAULT_MAX_RECENT_CHARS`].
    pub fn with_max_recent_chars(mut self, max_recent_chars: usize) -> Self {
        self.max_recent_chars = max_recent_chars;
        self
    }

    /// Filter text from a sliding window transcription.
    ///
    /// Finds where the previous output ends within the new transcription
    /// (anchor search) and returns only the text after that point. Stores
    /// the full new transcription as the reference for the next window.
    pub fn filter_text(&mut self, new_text: &str) -> String {
        let result = if self.recent_text.is_empty() {
            new_text.to_string()
        } else {
            remove_overlap(&self.recent_text, new_text)
        };

        // Store the full new transcription as reference for the next window.
        // Each window's complete output is what the next window will overlap with.
        self.recent_text = new_text.to_string();
        if self.recent_text.len() > self.max_recent_chars {
            // Trim to the last `max_recent_chars` bytes, but only at a UTF-8
            // char boundary so we never split a multi-byte codepoint.
            let target = self.recent_text.len() - self.max_recent_chars;
            let trim_at = floor_char_boundary(&self.recent_text, target);
            self.recent_text = self.recent_text[trim_at..].to_string();
        }

        result
    }
}

impl Default for TextDedup {
    fn default() -> Self {
        Self::new()
    }
}

/// Round `index` down to the nearest UTF-8 char boundary in `s`.
///
/// Equivalent to the unstable `str::floor_char_boundary` (tracking issue #93743).
/// Replace with that once it stabilizes.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Remove overlapping text between the end of `previous` and `new`.
///
/// Uses three strategies (tried in order):
///
/// 1. Anchor search — takes the last N words of `previous` and searches
///    for that sequence anywhere in `new`.  Handles whisper inserting or
///    deleting words anywhere in the overlapping region.  Searches the
///    entire `new` text (no percentage limit) since the anchor from the
///    end of `previous` will never appear at the very end of `new` for
///    legitimate new content.
///
/// 2. Longest common prefix (fuzzy fallback).  If anchor search finds
///    nothing, try matching from the start — whisper.cpp is explicitly
///    prompted with the previous text, so the new output often starts
///    with similar words.
///
/// 3. Suffix-prefix alignment — checks whether the end of `previous`
///    matches the start of `new`.
///
/// Falls back to returning the full `new` text if no overlap is found.
pub(crate) fn remove_overlap(previous: &str, new: &str) -> String {
    let prev_words: Vec<&str> = previous.split_whitespace().collect();
    let new_words: Vec<&str> = new.split_whitespace().collect();

    if prev_words.is_empty() || new_words.is_empty() {
        return new.to_string();
    }

    // --- Strategy 1: Anchor search ---
    // Take the last N words of previous and search for them in the new text.
    // Search the entire new text — if the anchor from the end of `previous`
    // appears, the text after it is genuinely new.
    let max_anchor = prev_words.len().min(8);

    for anchor_len in (3..=max_anchor).rev() {
        let anchor = &prev_words[prev_words.len() - anchor_len..];

        for pos in 0..new_words.len() {
            if pos + anchor_len > new_words.len() {
                break;
            }
            let candidate = &new_words[pos..pos + anchor_len];
            if ngram_match(anchor, candidate) {
                let new_start = pos + anchor_len;
                if new_start >= new_words.len() {
                    return String::new();
                }
                return new_words[new_start..].join(" ");
            }
        }
    }

    // --- Strategy 2: Longest common prefix (fuzzy fallback) ---
    // Because whisper is explicitly prompted with the previous output,
    // the start of the new text often repeats the previous text verbatim.
    let max_prefix = prev_words.len().min(new_words.len()).min(50);
    let mut prefix_len = 0;
    for i in 0..max_prefix {
        if !words_match(prev_words[i], new_words[i]) {
            break;
        }
        prefix_len = i + 1;
    }
    if prefix_len > 0 {
        let remaining = &new_words[prefix_len..];
        if remaining.is_empty() {
            return String::new();
        }
        return remaining.join(" ");
    }

    // --- Strategy 3: Suffix-prefix alignment ---
    // Check if the end of previous matches the start of new.
    let max_overlap = prev_words.len().min(new_words.len()).min(50);
    for overlap_len in (1..=max_overlap).rev() {
        let prev_suffix = &prev_words[prev_words.len() - overlap_len..];
        let new_prefix = &new_words[..overlap_len];

        if ngram_match(prev_suffix, new_prefix) {
            let remaining = &new_words[overlap_len..];
            if remaining.is_empty() {
                return String::new();
            }
            return remaining.join(" ");
        }
    }

    // No overlap found — return the full new text.
    new.to_string()
}

/// Check if two word slices match (allowing fuzzy matching per word).
pub(crate) fn ngram_match(a: &[&str], b: &[&str]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(wa, wb)| words_match(wa, wb))
}

/// Check if two words match, allowing for minor differences in punctuation
/// and small edit distances.
pub(crate) fn words_match(a: &str, b: &str) -> bool {
    // Normalize: lowercase and strip trailing punctuation.
    let na = normalize_word(a);
    let nb = normalize_word(b);

    if na == nb {
        return true;
    }

    // Use Jaro-Winkler similarity for fuzzy matching.
    let similarity = strsim::jaro_winkler(&na, &nb);
    similarity >= 0.85
}

/// Normalize a word for comparison: lowercase, strip trailing punctuation.
pub(crate) fn normalize_word(word: &str) -> String {
    word.to_lowercase()
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_dedup_no_overlap() {
        let mut tracker = TimestampDedup::new();

        let words = vec![
            Word {
                text: "Hello".to_string(),
                start_secs: 0.0,
                end_secs: 0.5,
            },
            Word {
                text: "world".to_string(),
                start_secs: 0.6,
                end_secs: 1.0,
            },
        ];

        let accepted = tracker.filter_words(&words);
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[0].text, "Hello");
        assert_eq!(accepted[1].text, "world");
    }

    #[test]
    fn timestamp_dedup_skips_overlapping() {
        let mut tracker = TimestampDedup::new();

        // First chunk.
        let words1 = vec![
            Word {
                text: "Hello".to_string(),
                start_secs: 0.0,
                end_secs: 0.5,
            },
            Word {
                text: "world".to_string(),
                start_secs: 0.6,
                end_secs: 1.0,
            },
        ];
        let accepted1 = tracker.filter_words(&words1);
        assert_eq!(accepted1.len(), 2);

        // Second chunk with overlap — these words start before transcribed_up_to.
        let words2 = vec![
            Word {
                text: "world".to_string(),
                start_secs: 0.6,
                end_secs: 1.0,
            },
            Word {
                text: "how".to_string(),
                start_secs: 1.1,
                end_secs: 1.3,
            },
        ];
        // No offset advance — simulate overlap.
        let accepted2 = tracker.filter_words(&words2);
        assert_eq!(accepted2.len(), 1);
        assert_eq!(accepted2[0].text, "how");
    }

    #[test]
    fn timestamp_dedup_with_offset() {
        let mut tracker = TimestampDedup::new();

        let words1 = vec![Word {
            text: "Hello".to_string(),
            start_secs: 0.0,
            end_secs: 0.5,
        }];
        tracker.filter_words(&words1);

        // Advance offset by 1 second (first chunk was 1s of audio).
        tracker.advance_offset(1.0);

        // Second chunk: timestamps restart from 0 but offset adjusts them.
        let words2 = vec![Word {
            text: "world".to_string(),
            start_secs: 0.1,
            end_secs: 0.5,
        }];
        let accepted = tracker.filter_words(&words2);
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0].text, "world");
        // Adjusted start should be ~1.1.
        assert!((accepted[0].start_secs - 1.1).abs() < 0.01);
    }

    #[test]
    fn timestamp_dedup_custom_overlap_tolerance() {
        // With a tight tolerance of 0.001s, a word that starts exactly 0.01s
        // before `transcribed_up_to` should be rejected — whereas the default
        // tolerance of 0.05s would accept it.
        let mut tracker = TimestampDedup::new().with_overlap_tolerance_secs(0.001);

        // Establish transcribed_up_to = 1.0.
        let words1 = vec![Word {
            text: "first".to_string(),
            start_secs: 0.0,
            end_secs: 1.0,
        }];
        let accepted1 = tracker.filter_words(&words1);
        assert_eq!(accepted1.len(), 1);

        // This word's start_secs (0.99) is 0.01s under transcribed_up_to (1.0).
        // With the default 0.05s tolerance it would be accepted; with our
        // 0.001s tolerance it must be rejected.
        let words2 = vec![Word {
            text: "second".to_string(),
            start_secs: 0.99,
            end_secs: 1.5,
        }];
        let accepted2 = tracker.filter_words(&words2);
        assert!(
            accepted2.is_empty(),
            "expected word to be rejected with tight tolerance, got {accepted2:?}"
        );
    }

    #[test]
    fn text_dedup_no_previous() {
        let mut tracker = TextDedup::new();
        let result = tracker.filter_text("Hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn text_dedup_removes_overlap_prefix() {
        // Simulates sliding window: window 2 re-transcribes window 1 + new text.
        let mut tracker = TextDedup::new();
        tracker.filter_text("the quick brown fox");
        let result = tracker.filter_text("the quick brown fox jumps over");
        assert_eq!(result, "jumps over");
    }

    #[test]
    fn text_dedup_handles_whisper_rephrase() {
        // Whisper changes "to see" → "and see" in the overlap region.
        // The anchor (last 3+ words of prev) should still find the match.
        let mut tracker = TextDedup::new();
        tracker.filter_text("trying to test it to see if it works");
        // Window 2 rephrased slightly but the end of prev ("if it works") is intact.
        let result =
            tracker.filter_text("trying to test it and see if it works right now I am speaking");
        assert_eq!(result, "right now I am speaking");
    }

    #[test]
    fn text_dedup_no_overlap_found() {
        let mut tracker = TextDedup::new();
        tracker.filter_text("Hello world");
        let result = tracker.filter_text("completely different text");
        assert_eq!(result, "completely different text");
    }

    #[test]
    fn text_dedup_full_overlap() {
        let mut tracker = TextDedup::new();
        tracker.filter_text("Hello world foo bar baz");
        let result = tracker.filter_text("Hello world foo bar baz");
        assert_eq!(result, "");
    }

    #[test]
    fn text_dedup_sliding_window_sequence() {
        // Simulate 3 overlapping windows.
        let mut tracker = TextDedup::new();

        let r1 = tracker.filter_text("A B C D E F");
        assert_eq!(r1, "A B C D E F");

        let r2 = tracker.filter_text("A B C D E F G H I");
        assert_eq!(r2, "G H I");

        let r3 = tracker.filter_text("D E F G H I J K L");
        assert_eq!(r3, "J K L");
    }

    #[test]
    fn text_dedup_custom_max_recent_chars() {
        // Configure a tiny buffer (16 bytes) and confirm the recent_text is
        // trimmed to at most that many bytes after a long input.
        let mut tracker = TextDedup::new().with_max_recent_chars(16);
        tracker.filter_text("the quick brown fox jumps over the lazy dog");
        assert!(
            tracker.recent_text.len() <= 16,
            "expected trim to ≤16 bytes, got {} bytes ({:?})",
            tracker.recent_text.len(),
            tracker.recent_text
        );
    }

    #[test]
    fn text_dedup_trim_respects_utf8_boundary() {
        // Pad with multi-byte characters so the naive byte-cut would land
        // inside a codepoint. Each "é" is 2 bytes, "中" is 3 bytes.
        // The buffer below is well over 32 bytes; with the OLD implementation
        // (raw byte slice `recent_text[trim_at..]`) trimming to 32 bytes
        // would split a multibyte char and panic.
        let mut tracker = TextDedup::new().with_max_recent_chars(32);
        let input = "é中é中é中é中é中é中é中é中é中é中é中é中é中é中é中é中";
        // Sanity: our input is non-trivially long.
        assert!(input.len() > 32);
        // This must NOT panic. (The naive `[trim_at..]` would have.)
        let _ = tracker.filter_text(input);
        // We floor `trim_at` down to a char boundary, so the surviving buffer
        // is at least max_recent_chars bytes and never more than
        // max_recent_chars + (max UTF-8 char len - 1) = max + 3.
        let len = tracker.recent_text.len();
        assert!(
            (32..=32 + 3).contains(&len),
            "expected trimmed buffer in 32..=35 bytes, got {len} ({:?})",
            tracker.recent_text
        );
        // The surviving String must be valid UTF-8 with a clean prefix —
        // re-decoding from its bytes must round-trip.
        let bytes = tracker.recent_text.as_bytes().to_vec();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), tracker.recent_text);
    }

    #[test]
    fn normalize_word_strips_punctuation() {
        assert_eq!(normalize_word("Hello,"), "hello");
        assert_eq!(normalize_word("world."), "world");
        assert_eq!(normalize_word("test"), "test");
    }

    #[test]
    fn words_match_exact() {
        assert!(words_match("hello", "hello"));
        assert!(words_match("Hello", "hello"));
    }

    #[test]
    fn words_match_with_punctuation() {
        assert!(words_match("hello,", "hello"));
        assert!(words_match("world.", "world"));
    }

    #[test]
    fn words_match_fuzzy() {
        // Small edit distance should still match.
        assert!(words_match("hello", "helo"));
    }

    #[test]
    fn words_dont_match_very_different() {
        assert!(!words_match("hello", "world"));
    }

    #[test]
    fn remove_overlap_anchor_search() {
        // Anchor "three four" from end of prev, found in new text.
        let result = remove_overlap("one two three four", "two three four five six");
        assert_eq!(result, "five six");
    }

    #[test]
    fn remove_overlap_with_inserted_word() {
        // Whisper inserted "really" but the end anchor still matches.
        let result = remove_overlap(
            "I think it is going to work",
            "I really think it is going to work now",
        );
        assert_eq!(result, "now");
    }

    #[test]
    fn remove_overlap_none() {
        let result = remove_overlap("hello world", "completely different");
        assert_eq!(result, "completely different");
    }

    #[test]
    fn remove_overlap_prefix_fallback() {
        // Short prev — falls back to prefix alignment.
        let result = remove_overlap("brown fox", "brown fox jumps");
        assert_eq!(result, "jumps");
    }

    #[test]
    fn floor_char_boundary_handles_multibyte() {
        let s = "é中é"; // 2 + 3 + 2 = 7 bytes
                        // Index 1 is inside "é" (which spans bytes 0..2) → floor to 0.
        assert_eq!(floor_char_boundary(s, 1), 0);
        // Index 2 is the start of "中" → already a boundary.
        assert_eq!(floor_char_boundary(s, 2), 2);
        // Index 3 is inside "中" (which spans bytes 2..5) → floor to 2.
        assert_eq!(floor_char_boundary(s, 3), 2);
        // Index 4 is also inside "中" → floor to 2.
        assert_eq!(floor_char_boundary(s, 4), 2);
        // Index 5 is the start of "é" → boundary.
        assert_eq!(floor_char_boundary(s, 5), 5);
        // Index past the end clamps to len.
        assert_eq!(floor_char_boundary(s, 100), s.len());
    }
}
