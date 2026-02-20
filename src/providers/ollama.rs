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
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
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
            r#"Fix grammar and structure this journal entry. Return JSON with title, content, and tags.

Input: {}

CRITICAL RULES:
- NEVER translate the text - keep the EXACT same language as the input
- NEVER add new information or content not in the original
- ONLY fix spelling mistakes and grammar errors
- ONLY improve sentence structure and formatting
- Keep ALL original meaning and content intact
- Title: 3-5 words describing the note, lowercase, hyphen-separated, ends with .md
- Content: cleaned up version of the input with better formatting (paragraphs, bullet points if needed)
- Tags: 0-3 relevant keywords from the content

Return ONLY valid JSON:
{{"title": "short-descriptive-name.md", "content": "Cleaned up content here", "tags": ["tag1", "tag2"]}}"#,
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
            options: Some(OllamaOptions { temperature: 0.1 }),
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
        // Try to check if Ollama is running by making a simple request
        // Use a blocking reqwest client for the check
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/api/tags", self.config.base_url);
        
        match client.get(&url).timeout(std::time::Duration::from_secs(2)).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let prompt = OllamaProvider::build_prompt("Meeting with team");
        assert!(prompt.contains("Fix grammar"));
        assert!(prompt.contains("Meeting with team"));
        assert!(prompt.contains("JSON"));
        assert!(prompt.contains("NEVER translate"));
    }
}
