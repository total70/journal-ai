use crate::config::OpenAiConfig;
use crate::providers::{sanitize_title, LlmProvider, LlmResponse};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OpenAiProvider {
    config: OpenAiConfig,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    type_: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: String,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self> {
        if config.api_key.is_none() {
            return Err(anyhow!("OpenAI API key not configured. Set OPENAI_API_KEY environment variable or add to config"));
        }
        
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn build_messages(user_input: &str, system_prompt: Option<&str>) -> Vec<Message> {
        let system_content = system_prompt.unwrap_or(
            "You are a journal assistant. Structure entries and return JSON with title, content, and tags. \
            Title rules: 3-5 words, lowercase, hyphen-separated, ends with .md. \
            Return ONLY valid JSON."
        );

        vec![
            Message {
                role: "system".to_string(),
                content: system_content.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: format!(
                    r#"Structure this journal entry and return JSON:

Input: {}

Return ONLY valid JSON:
{{"title": "short-descriptive-name.md", "content": "Structured content here", "tags": ["tag1", "tag2"]}}"#,
                    user_input
                ),
            },
        ]
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<LlmResponse> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| anyhow!("OpenAI API key not set"))?;

        let messages = Self::build_messages(prompt, system_prompt);

        let request = OpenAiRequest {
            model: self.config.model.clone(),
            messages,
            temperature: 0.3, // Lower temperature for more consistent formatting
            response_format: Some(ResponseFormat {
                type_: "json_object".to_string(),
            }),
        };

        let url = format!("{}/chat/completions", self.config.base_url);
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to connect to OpenAI API")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("OpenAI API error {}: {}", status, text));
        }

        let openai_resp: OpenAiResponse = response
            .json()
            .await
            .context("Failed to parse OpenAI response")?;

        let content = openai_resp.choices
            .first()
            .ok_or_else(|| anyhow!("No response from OpenAI"))?
            .message
            .content
            .clone();

        // Parse the JSON response
        let llm_response: LlmResponse = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse LLM JSON response: {}", content))?;

        // Sanitize the title
        let title = sanitize_title(&llm_response.title);

        Ok(LlmResponse {
            title,
            content: llm_response.content,
            tags: llm_response.tags,
        })
    }

    fn is_available(&self) -> bool {
        self.config.api_key.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_messages() {
        let messages = OpenAiProvider::build_messages("Test input", None);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert!(messages[1].content.contains("Test input"));
    }

    #[test]
    fn test_custom_system_prompt() {
        let messages = OpenAiProvider::build_messages("Test", Some("Custom prompt"));
        assert_eq!(messages[0].content, "Custom prompt");
    }
}
