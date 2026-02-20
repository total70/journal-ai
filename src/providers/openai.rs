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
            "You ONLY fix grammar and formatting. \
            NEVER translate. NEVER add commentary like 'here is' or summaries. \
            NEVER add content not in original. \
            Output ONLY cleaned text, nothing else."
        );

        vec![
            Message {
                role: "system".to_string(),
                content: system_content.to_string(),
            },
            Message {
                role: "user".to_string(),
                content: format!(
                    r#"Clean up this text. Fix spelling/grammar only.

Input: {}

RULES:
- Same language as input
- NO added commentary or explanations
- NO "here is" or "summary" text
- NO new information
- ONLY fix errors and formatting

Return JSON:
{{"title": "name.md", "content": "cleaned text only", "tags": []}}"#,
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
            temperature: 0.1, // Very low temperature for less creativity, more consistency
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

    async fn summarize(&self, prompt: &str) -> Result<String> {
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| anyhow!("OpenAI API key not set"))?;

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant that summarizes journal entries. Be concise and highlight key points.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];

        let request = OpenAiRequest {
            model: self.config.model.clone(),
            messages,
            temperature: 0.3,
            response_format: None, // No JSON mode for summarize
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

        Ok(openai_resp.choices
            .first()
            .ok_or_else(|| anyhow!("No response from OpenAI"))?
            .message
            .content
            .clone())
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
