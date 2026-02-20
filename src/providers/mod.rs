use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod ollama;
pub mod openai;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<LlmResponse>;
    async fn summarize(&self, prompt: &str) -> Result<String>;
    fn is_available(&self) -> bool;
}

/// Sanitize title to be filesystem-safe
pub fn sanitize_title(title: &str) -> String {
    let mut safe = title
        .replace(' ', "-")
        .replace('/', "-")
        .replace('\\', "-")
        .replace(':', "-")
        .replace('?', "-")
        .replace('*', "-")
        .replace('"', "-")
        .replace('\'', "-")
        .replace('<', "-")
        .replace('>', "-")
        .replace('|', "-")
        .to_lowercase();

    // Collapse multiple hyphens
    while safe.contains("--") {
        safe = safe.replace("--", "-");
    }

    // Trim trailing hyphen and whitespace
    safe = safe.trim_end_matches('-').trim().to_string();

    // Ensure it ends with .md
    if !safe.ends_with(".md") {
        safe.push_str(".md");
    }

    safe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_title_basic() {
        assert_eq!(sanitize_title("Hello World"), "hello-world.md");
    }

    #[test]
    fn test_sanitize_title_with_punctuation() {
        assert_eq!(sanitize_title("test: file/name"), "test-file-name.md");
    }

    #[test]
    fn test_sanitize_title_multiple_hyphens() {
        assert_eq!(sanitize_title("my---daily---notes"), "my-daily-notes.md");
    }

    #[test]
    fn test_sanitize_title_trailing_hyphen() {
        assert_eq!(sanitize_title("trailing?"), "trailing.md");
    }

    #[test]
    fn test_sanitize_title_already_has_md() {
        assert_eq!(sanitize_title("already.md"), "already.md");
    }

    #[test]
    fn test_sanitize_title_mixed_case() {
        assert_eq!(sanitize_title("Meeting With TEAM"), "meeting-with-team.md");
    }
}
