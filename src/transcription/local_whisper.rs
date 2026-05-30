//! Local whisper.cpp transcription backend via `whisper-rs`.
//!
//! Uses whisper-rs with a sliding window approach for pseudo-streaming.
//!
//! Deduplication strategy: each window is transcribed with `set_initial_prompt()`
//! set to the previous output for consistency. Text-based n-gram overlap removal
//! (from `dedup.rs`) strips the repeated prefix. Timestamp filtering is not used
//! because whisper often produces a single segment with `start_timestamp=0`,
//! making timestamp-based approaches unreliable.

use std::sync::Arc;

use asr_dedup::remove_overlap;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::{TranscriptionBackend, TranscriptionConfig};
use crate::audio::AudioChunk;

/// Sliding window parameters for pseudo-streaming.
const WINDOW_SECS: usize = 8;
const STEP_SECS: usize = 2;
const SAMPLE_RATE: usize = 16_000;
const WINDOW_SAMPLES: usize = WINDOW_SECS * SAMPLE_RATE;
const STEP_SAMPLES: usize = STEP_SECS * SAMPLE_RATE;
/// Shorter initial window for faster first result.
const INITIAL_WINDOW_SECS: usize = 4;
const INITIAL_WINDOW_SAMPLES: usize = INITIAL_WINDOW_SECS * SAMPLE_RATE;

/// Maximum prompt length (chars) fed to whisper as `initial_prompt`.
/// Longer prompts cause whisper to "predict" upcoming words instead of
/// transcribing from audio. ~80 chars gives enough context for consistency
/// without enabling prediction.
const MAX_PROMPT_CHARS: usize = 80;

/// Silence threshold — must match or be below the daemon's auto-stop threshold
/// (0.003) so we never skip windows that auto-stop considers speech.
const SILENCE_THRESHOLD: f64 = 0.003;

/// Local whisper.cpp transcription backend.
pub struct LocalWhisperBackend {
    ctx: Option<Arc<whisper_rs::WhisperContext>>,
    #[allow(dead_code)]
    model_path: String,
}

impl LocalWhisperBackend {
    /// Create a new local whisper backend, eagerly loading the model.
    pub fn new(model_path: String) -> Self {
        let ctx = match Self::load_model(&model_path) {
            Ok(ctx) => {
                info!("loaded whisper model from {model_path}");
                Some(Arc::new(ctx))
            }
            Err(e) => {
                warn!("failed to load whisper model from {model_path}: {e}");
                None
            }
        };
        Self { ctx, model_path }
    }

    fn load_model(path: &str) -> anyhow::Result<whisper_rs::WhisperContext> {
        if !std::path::Path::new(path).exists() {
            anyhow::bail!("model file not found: {path}. Run 'whisrs setup' to download a model.");
        }

        let params = whisper_rs::WhisperContextParameters::default();

        whisper_rs::WhisperContext::new_with_params(path, params)
            .map_err(|e| anyhow::anyhow!("failed to initialize whisper context: {e}"))
    }
}

/// Strip hallucination tokens that whisper.cpp models sometimes emit for
/// silence and non-speech audio (e.g. `[BLANK_AUDIO]`, `[COUGH]`). These
/// degrade dedup accuracy and pollute the transcription.
fn strip_whisper_tokens(text: &str) -> String {
    /// Known whisper hallucination tokens (handled case-insensitively).
    const TOKENS: &[&str] = &[
        "[BLANK_AUDIO]",
        "[SILENCE]",
        "[INAUDIBLE]",
        "[MUSIC]",
        "[NOISE]",
        "[COUGH]",
        "[SNEEZE]",
        "[LAUGH]",
        "[APPLAUSE]",
        "[NO_SPEECH]",
        "[UNKNOWN]",
        "[STATIC]",
        "[BEEP]",
        "[SPEECH]",
    ];

    let mut result = text.to_string();
    for token in TOKENS {
        result = result.replace(&token.to_uppercase(), " ");
        if *token != token.to_lowercase() {
            result = result.replace(&token.to_lowercase(), " ");
        }
    }

    result
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Return the last `max_chars` characters of `s`, rounding down to the nearest
/// UTF-8 char boundary so we never split a multi-byte codepoint.
fn truncate_tail(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }
    let target = s.len() - max_chars;
    let mut i = target;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    &s[i..]
}

/// Convert i16 PCM samples to f32 in the range [-1.0, 1.0].
#[cfg(any(feature = "local-whisper", test))]
fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

/// Run whisper inference on an audio window.
///
/// - `prompt`: previous transcription to condition this window for consistency.
///   Whisper uses it to maintain context across overlapping windows.
fn run_whisper_inference(
    ctx: &whisper_rs::WhisperContext,
    audio: &[f32],
    language: &str,
    prompt: Option<&str>,
) -> anyhow::Result<String> {
    use whisper_rs::{FullParams, SamplingStrategy};

    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    if language != "auto" {
        params.set_language(Some(language));
    }

    // Feed previous transcription as prompt so whisper produces consistent
    // output across overlapping windows.
    if let Some(prompt) = prompt {
        if !prompt.is_empty() {
            params.set_initial_prompt(prompt);
        }
    }
    // Allow whisper to use the prompt as context.
    params.set_no_context(false);

    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    params.set_n_threads(n_threads);

    state
        .full(params, audio)
        .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;

    let mut text = String::new();
    for segment in state.as_iter() {
        text.push_str(&format!("{}", segment));
    }

    Ok(text.trim().to_string())
}

#[async_trait]
impl TranscriptionBackend for LocalWhisperBackend {
    async fn transcribe(
        &self,
        audio: &[u8],
        config: &TranscriptionConfig,
    ) -> anyhow::Result<String> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "whisper model not loaded from {}. Run 'whisrs setup' to download a model.",
                self.model_path
            )
        })?;

        // Decode WAV to i16 samples, then convert to f32.
        let cursor = std::io::Cursor::new(audio);
        let reader = hound::WavReader::new(cursor)?;
        let samples_i16: Vec<i16> = reader.into_samples::<i16>().collect::<Result<_, _>>()?;
        let mut samples_f32 = vec![0.0f32; samples_i16.len()];
        whisper_rs::convert_integer_to_float_audio(&samples_i16, &mut samples_f32)
            .map_err(|e| anyhow::anyhow!("failed to convert audio: {e}"))?;

        let ctx = Arc::clone(ctx);
        let language = config.language.clone();
        let prompt = config.prompt.clone();

        tokio::task::spawn_blocking(move || {
            run_whisper_inference(&ctx, &samples_f32, &language, prompt.as_deref())
        })
        .await?
    }

    async fn transcribe_stream(
        &self,
        mut audio_rx: mpsc::Receiver<AudioChunk>,
        text_tx: mpsc::Sender<String>,
        config: &TranscriptionConfig,
    ) -> anyhow::Result<()> {
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("whisper model not loaded from {}", self.model_path))?;

        let mut buffer: Vec<i16> = Vec::new();
        let mut running_transcript = String::new();
        let mut next_process_at = INITIAL_WINDOW_SAMPLES;
        let mut last_processed_end: usize = 0;
        // Short prompt for whisper to prevent prediction. ~80 chars gives
        // enough context for consistent output without hallucinating ahead.
        let mut prompt = truncate_tail(
            config.prompt.as_deref().unwrap_or(""),
            MAX_PROMPT_CHARS,
        )
        .to_string();

        while let Some(chunk) = audio_rx.recv().await {
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= next_process_at {
                let window_size = if last_processed_end == 0 {
                    INITIAL_WINDOW_SAMPLES.min(buffer.len())
                } else {
                    WINDOW_SAMPLES.min(buffer.len())
                };

                let window_end = next_process_at;
                let window_start = window_end.saturating_sub(window_size);
                let window = buffer[window_start..window_end].to_vec();

                // Skip silent windows.
                if !audio_silence_gate::is_silent(&window, SILENCE_THRESHOLD) {
                    let samples_f32 = i16_to_f32(&window);
                    let ctx_clone = Arc::clone(ctx);
                    let lang = config.language.clone();
                    let prev_prompt = if prompt.is_empty() {
                        None
                    } else {
                        Some(prompt.clone())
                    };

                    match tokio::task::spawn_blocking(move || {
                        run_whisper_inference(
                            &ctx_clone,
                            &samples_f32,
                            &lang,
                            prev_prompt.as_deref(),
                        )
                    })
                    .await
                    {
                        Ok(Ok(full_text)) => {
                            let stripped = strip_whisper_tokens(&full_text);
                            if !stripped.is_empty() {
                                prompt = truncate_tail(&stripped, MAX_PROMPT_CHARS).to_string();
                            }
                            let new_text = if running_transcript.is_empty() {
                                stripped.clone()
                            } else {
                                remove_overlap(&running_transcript, &stripped)
                            };
                            if !new_text.trim().is_empty() {
                                debug!("streaming window produced: {:?}", new_text);
                                if !running_transcript.is_empty() {
                                    running_transcript.push(' ');
                                }
                                running_transcript.push_str(&new_text);
                                text_tx.send(new_text).await.ok();
                            }
                        }
                        Ok(Err(e)) => warn!("whisper window inference failed: {e}"),
                        Err(e) => warn!("whisper task panicked: {e}"),
                    }
                } else {
                    debug!(
                        "skipping silent window at samples {}..{}",
                        window_start, window_end
                    );
                }

                last_processed_end = window_end;
                next_process_at += STEP_SAMPLES;
            }
        }

        // Process remaining audio not covered by the last window.
        if buffer.len() > last_processed_end {
            let remaining_start = if buffer.len() - last_processed_end < SAMPLE_RATE {
                last_processed_end.saturating_sub(WINDOW_SAMPLES / 4)
            } else {
                last_processed_end
            };
            let remaining = &buffer[remaining_start..];

            if !remaining.is_empty() && !audio_silence_gate::is_silent(remaining, SILENCE_THRESHOLD)
            {
                let samples_f32 = i16_to_f32(remaining);
                let ctx_clone = Arc::clone(ctx);
                let lang = config.language.clone();
                let prev_prompt = if prompt.is_empty() {
                    None
                } else {
                    Some(prompt.clone())
                };

                match tokio::task::spawn_blocking(move || {
                    run_whisper_inference(&ctx_clone, &samples_f32, &lang, prev_prompt.as_deref())
                })
                .await
                {
                    Ok(Ok(full_text)) => {
                        let stripped = strip_whisper_tokens(&full_text);
                        let new_text = if running_transcript.is_empty() {
                            stripped.clone()
                        } else {
                            remove_overlap(&running_transcript, &stripped)
                        };
                        if !new_text.trim().is_empty() {
                            if !running_transcript.is_empty() {
                                running_transcript.push(' ');
                            }
                            running_transcript.push_str(&new_text);
                            text_tx.send(new_text).await.ok();
                        }
                    }
                    Ok(Err(e)) => warn!("whisper final inference failed: {e}"),
                    Err(e) => warn!("whisper final task panicked: {e}"),
                }
            }
        }

        Ok(())
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_returns_error() {
        let backend = LocalWhisperBackend::new("/nonexistent/model.bin".to_string());
        let config = TranscriptionConfig {
            language: "en".to_string(),
            model: "base.en".to_string(),
            prompt: None,
        };
        let err = backend.transcribe(&[1, 2, 3], &config).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not available")
                || msg.contains("not loaded")
                || msg.contains("not found"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn i16_to_f32_conversion() {
        let samples = vec![0i16, i16::MAX, i16::MIN];
        let f32_samples = i16_to_f32(&samples);
        assert_eq!(f32_samples[0], 0.0);
        assert!((f32_samples[1] - 1.0).abs() < 0.001);
        assert!((f32_samples[2] + 1.0).abs() < 0.001);
    }
}
