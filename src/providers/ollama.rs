use crate::config::OllamaConfig;
use crate::providers::{sanitize_title, LlmProvider, LlmResponse, TaskItem};
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

/// Extract JSON object from a string that may contain markdown code blocks or extra text
fn extract_json(raw: &str) -> String {
    // Try to find JSON between ```json ... ``` or ``` ... ```
    if let Some(start) = raw.find("```json") {
        if let Some(end) = raw[start + 7..].find("```") {
            return raw[start + 7..start + 7 + end].trim().to_string();
        }
    }
    if let Some(start) = raw.find("```") {
        if let Some(end) = raw[start + 3..].find("```") {
            return raw[start + 3..start + 3 + end].trim().to_string();
        }
    }
    // Try to find the first { ... } block
    if let Some(start) = raw.find('{') {
        if let Some(end) = raw.rfind('}') {
            if end > start {
                return raw[start..=end].to_string();
            }
        }
    }
    // Fallback: return as-is
    raw.trim().to_string()
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
            r#"Fix grammar and structure this journal entry. Return JSON.

Input: {}

ABSOLUTE RULES - VIOLATION IS NOT ALLOWED:
1. NEVER translate - keep EXACT same language as input
2. NEVER add explanations, summaries, or meta-text like "here is", "note that", "summary of"
3. NEVER add content not present in the original input
4. NEVER describe what you did or add commentary
5. Output ONLY the cleaned content, nothing else

ALLOWED changes:
- Fix spelling errors
- Fix grammar mistakes  
- Add punctuation
- Split into paragraphs or bullet points for readability

Title: 3-5 words, lowercase, hyphen-separated, ends with .md
Content: cleaned content ONLY, no added commentary
Tags: 0-3 keywords from content

Return ONLY this JSON:
{{
  "title": "name.md",
  "content": "cleaned content",
  "tags": ["tag1"],
  "tasks": [
    {{"text": "Call Jan about Q2 planning", "priority": "normal", "due": null}}
  ]
}}

Tasks rules:
- Extract explicit action items and to-dos from the input
- Only include tasks that are clearly actionable
- Keep task text short (1 sentence)
- priority must be one of: low, normal, high
- due must be null or ISO date string (YYYY-MM-DD)
- If no tasks, return an empty array for tasks
"#,
            user_input
        )
    }

    fn build_tasks_prompt(clean_content: &str) -> String {
        format!(
            r#"Extract actionable tasks from the following note content. Return ONLY JSON.

Content:
{}

Return ONLY this JSON:
{{
  \"tasks\": [
    {{\"text\": \"Call Jan about Q2 planning\", \"priority\": \"normal\", \"due\": null}}
  ]
}}

Rules:
- Extract explicit action items / to-dos
- Also treat scheduled plans as tasks (dates, weekdays, "tomorrow", "next week", etc.), even if not written as an imperative
- Task text MUST be in the same language as the content (never translate)
- If no tasks, return: {{\"tasks\": []}}
- Keep task text short (1 sentence)
- priority must be one of: low, normal, high (default normal)
- due must be null or ISO date string (YYYY-MM-DD)
"#,
            clean_content
        )
    }

    async fn call_ollama_json(&self, prompt: &str, system_prompt: Option<&str>) -> Result<String> {
        let request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            system: system_prompt.map(|s| s.to_string()),
            stream: false,
            format: Some("json".to_string()),
            options: Some(OllamaOptions { temperature: 0.1 }),
        };

        let url = format!("{}/api/generate", self.config.base_url);

        let response = self
            .client
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

        Ok(ollama_resp.response)
    }

    async fn generate_tasks(&self, clean_content: &str, system_prompt: Option<&str>) -> Result<Vec<TaskItem>> {
        #[derive(Deserialize)]
        struct TasksOnly {
            tasks: Option<Vec<TaskItem>>,
        }

        let tasks_prompt = Self::build_tasks_prompt(clean_content);
        let raw = self.call_ollama_json(&tasks_prompt, system_prompt).await?;
        let json_str = extract_json(&raw);

        let parsed: TasksOnly = serde_json::from_str(&json_str)
            .with_context(|| format!("Failed to parse tasks JSON response: {}", raw))?;

        Ok(parsed.tasks.unwrap_or_default())
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn generate(&self, prompt: &str, system_prompt: Option<&str>) -> Result<LlmResponse> {
        let full_prompt = Self::build_prompt(prompt);

        let raw = self.call_ollama_json(&full_prompt, system_prompt).await?;
        let json_str = extract_json(&raw);

        let llm_response: LlmResponse = serde_json::from_str(&json_str)
            .with_context(|| format!("Failed to parse LLM JSON response: {}", raw))?;

        fn strip_prompt_echo(s: &str) -> String {
            let markers = [
                "ABSOLUTE RULES",
                "Return ONLY this JSON",
                "Tasks rules:",
                "Title:",
                "Content:",
                "Tags:",
            ];

            let mut cut = s.len();
            for m in markers {
                if let Some(i) = s.find(m) {
                    cut = cut.min(i);
                }
            }

            s[..cut].trim().to_string()
        }

        let cleaned_content = strip_prompt_echo(&llm_response.content);

        // If we still see prompt instructions after stripping, fail loudly.
        if cleaned_content.contains("ABSOLUTE RULES") || cleaned_content.contains("Return ONLY this JSON") {
            return Err(anyhow!(
                "LLM returned prompt instructions as content — model may not support JSON mode. Raw response: {}",
                raw
            ));
        }

        // Sanitize the title
        let title = sanitize_title(&llm_response.title);

        fn has_time_signal(s: &str) -> bool {
            let s_l = s.to_lowercase();
            // numeric dates (language-agnostic)
            if s_l.contains('-') {
                // very cheap checks
                if s_l.chars().filter(|c| *c == '-').count() >= 2 {
                    return true;
                }
            }
            let needles = [
                "tomorrow",
                "next week",
                "next month",
                "next year",
                "next ",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ];
            needles.iter().any(|n| s_l.contains(n))
        }

        fn has_action_signal(s: &str) -> bool {
            let s_l = s.to_lowercase();
            let needles = [
                "review",
                "check",
                "test",
                "fix",
                "update",
                "revise",
                "refactor",
                "plan",
                "prepare",
                "herzien",
                "kijken of",
                "acceptatie",
                "criteria",
            ];
            needles.iter().any(|n| s_l.contains(n))
        }

        let mut tasks = match self.generate_tasks(&cleaned_content, system_prompt).await {
            Ok(t) => t,
            Err(_) => vec![],
        };

        // Conservative fallback: if the model returns 0 tasks but the note looks like scheduled work,
        // create a single task from the first non-empty line.
        if tasks.is_empty() && has_time_signal(&cleaned_content) && has_action_signal(&cleaned_content) {
            let first_line = cleaned_content
                .lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty() && !l.starts_with('#'))
                .unwrap_or(cleaned_content.trim());

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
            content: cleaned_content,
            tags: llm_response.tags,
            tasks,
        })
    }

    async fn summarize(&self, prompt: &str) -> Result<String> {
        let request = OllamaRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            system: Some("You are a helpful assistant that summarizes journal entries. Be concise and highlight key points. IMPORTANT: Always respond in the SAME language as the journal entries - never translate to another language.".to_string()),
            stream: false,
            format: None,
            options: Some(OllamaOptions { temperature: 0.3 }),
        };

        let url = format!("{}/api/generate", self.config.base_url);

        let response = self
            .client
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

        Ok(ollama_resp.response)
    }

    fn is_available(&self) -> bool {
        // Non-blocking availability check suitable for calling from within a Tokio runtime.
        // We simply attempt a short TCP connect to the host:port from base_url.
        use std::net::{TcpStream, ToSocketAddrs};
        use std::time::Duration;

        let base = self.config.base_url.trim();
        let base = base
            .strip_prefix("http://")
            .or_else(|| base.strip_prefix("https://"))
            .unwrap_or(base);
        let host_port = base.split('/').next().unwrap_or(base);

        let mut addrs_iter = match host_port.to_socket_addrs() {
            Ok(it) => it,
            Err(_) => return false,
        };

        if let Some(addr) = addrs_iter.next() {
            TcpStream::connect_timeout(&addr, Duration::from_secs(2)).is_ok()
        } else {
            false
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
        assert!(prompt.contains("NEVER"));
        assert!(prompt.contains("NO added commentary") || prompt.contains("commentary"));
    }
}
