use crate::config::OpenAiConfig;
use crate::providers::{sanitize_title, LlmProvider, LlmResponse, TaskItem};
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
            Output ONLY cleaned text, nothing else.",
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

Input: {input}

RULES:
- Same language as input (NEVER translate)
- NO added commentary or explanations
- NO "here is" or "summary" text
- NO new information
- ONLY fix errors and formatting

Return JSON with these exact fields:
- "title": 3-5 words from the content, lowercase, hyphen-separated, ends with .md (e.g. "call-jan-q2.md")
- "content": cleaned text only, same language as input
- "tags": 0-3 keywords
- "tasks": actionable items extracted from input

Tasks rules:
- Extract explicit action items and to-dos from the input
- Task text MUST be in the SAME language as the input (never translate)
- Only include tasks that are clearly actionable
- Keep task text short (1 sentence)
- priority must be one of: low, normal, high
- due must be null or ISO date string (YYYY-MM-DD)
- If no tasks, return an empty array

Return ONLY valid JSON, no markdown fences:
{{"title": "short-descriptive-title.md", "content": "...", "tags": [], "tasks": []}}
"#,
                    input = user_input
                ),
            },
        ]
    }

    fn has_time_signal(s: &str) -> bool {
        let s_l = s.to_lowercase();
        if s_l.chars().filter(|c| *c == '-').count() >= 2 {
            return true;
        }
        let needles = [
            "tomorrow", "next week", "next month", "next year", "next ",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "morgen", "volgende week", "maandag", "dinsdag", "woensdag", "donderdag",
            "vrijdag", "zaterdag", "zondag",
        ];
        needles.iter().any(|n| s_l.contains(n))
    }

    fn has_action_signal(s: &str) -> bool {
        let s_l = s.to_lowercase();
        let needles = [
            "review", "check", "test", "fix", "update", "revise", "refactor",
            "plan", "prepare", "call", "bel", "mail", "stuur", "maak", "schrijf",
            "afspraak", "vergadering", "meeting", "rapport", "report",
        ];
        needles.iter().any(|n| s_l.contains(n))
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<LlmResponse> {
        let api_key = self
            .config
            .api_key
            .as_ref()
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

        let response = self
            .client
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

        let content = openai_resp
            .choices
            .first()
            .ok_or_else(|| anyhow!("No response from OpenAI"))?
            .message
            .content
            .clone();

        // Strip markdown fences in case the model adds them anyway
        let json_str = {
            let trimmed = content.trim();
            if let Some(start) = trimmed.find("```json") {
                if let Some(end) = trimmed[start + 7..].find("```") {
                    trimmed[start + 7..start + 7 + end].trim().to_string()
                } else {
                    trimmed.to_string()
                }
            } else if let Some(start) = trimmed.find("```") {
                if let Some(end) = trimmed[start + 3..].find("```") {
                    trimmed[start + 3..start + 3 + end].trim().to_string()
                } else {
                    trimmed.to_string()
                }
            } else {
                trimmed.to_string()
            }
        };

        // Parse the JSON response
        let llm_response: LlmResponse = serde_json::from_str(&json_str)
            .with_context(|| format!("Failed to parse LLM JSON response: {}", content))?;

        // Sanitize the title; guard against the model returning the literal example placeholder
        let raw_title = &llm_response.title;
        let title = if raw_title.trim().is_empty()
            || raw_title == "name.md"
            || raw_title == "title.md"
            || raw_title == "short-descriptive-title.md"
        {
            // Derive from the first few words of the original prompt
            let words: Vec<&str> = prompt.split_whitespace().take(5).collect();
            sanitize_title(&words.join(" "))
        } else {
            sanitize_title(raw_title)
        };

        // Tasks fallback: if the model returned no tasks but the note looks like scheduled work
        let mut tasks = llm_response.tasks;
        if tasks.is_empty()
            && Self::has_time_signal(&llm_response.content)
            && Self::has_action_signal(&llm_response.content)
        {
            let first_line = llm_response.content
                .lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .unwrap_or(llm_response.content.trim());

            if !first_line.is_empty() {
                tasks.push(TaskItem {
                    text: first_line.to_string(),
                    priority: "normal".to_string(),
                    due: None,
                });
            }
        }

        Ok(LlmResponse {
            title,
            content: llm_response.content,
            tags: llm_response.tags,
            tasks,
        })
    }

    async fn summarize(&self, prompt: &str) -> Result<String> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| anyhow!("OpenAI API key not set"))?;

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant that summarizes journal entries. Be concise and highlight key points. IMPORTANT: Always respond in the SAME language as the journal entries - never translate to another language.".to_string(),
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

        let response = self
            .client
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

        Ok(openai_resp
            .choices
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
