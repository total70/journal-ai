use crate::config::OllamaConfig;
use crate::providers::{sanitize_title, LlmProvider, LlmResponse};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    system: Option<String>,
    stream: bool,
    format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    fn build_prompt(user_input: &str) -> String {
        format!(
            r#"Structure this journal entry and return JSON with title, content, and tags.

Input: {}

Rules:
- Title: 3-5 words, lowercase, hyphen-separated, ends with .md
- Content: structured version of the input, can be expanded slightly
- Tags: 0-3 relevant keywords

Return ONLY valid JSON in this format:
{{"title": "short-descriptive-name.md", "content": "Structured content here", "tags": ["tag1", "tag2"]}}"#,
            user_input
        )
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<LlmResponse> {
        let full_prompt = Self::build_prompt(prompt);
        
        let request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: full_prompt,
            system: system_prompt.map(|s| s.to_string()),
            stream: false,
            format: Some("json".to_string()),
        };

        let url = format!("{}/api/generate", self.config.base_url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .with_context(|| format!("Failed to connect to Ollama at {}", self.config.base_url))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama API error {}: {}", status, text));
        }

        let ollama_resp: OllamaResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        // Parse the JSON response from the LLM
        let llm_response: LlmResponse = serde_json::from_str(&ollama_resp.response)
            .with_context(|| format!("Failed to parse LLM JSON response: {}", ollama_resp.response))?;

        // Sanitize the title
        let title = sanitize_title(&llm_response.title);

        Ok(LlmResponse {
            title,
            content: llm_response.content,
            tags: llm_response.tags,
        })
    }

    fn is_available(&self) -> bool {
        // Try to check if Ollama is running
        let runtime = tokio::runtime::Handle::try_current();
        if runtime.is_err() {
            // We're not in an async context, can't check
            return true;
        }
        
        // In async context, we'd need to spawn a task
        // For now, assume available and let generate() fail if not
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let prompt = OllamaProvider::build_prompt("Meeting with team");
        assert!(prompt.contains("Structure this journal entry"));
        assert!(prompt.contains("Meeting with team"));
        assert!(prompt.contains("JSON"));
    }
}
