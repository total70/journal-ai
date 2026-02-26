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
- Only include explicit action items / to-dos
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

        // Validate: content should not contain the original prompt instructions
        if llm_response.content.contains("ABSOLUTE RULES")
            || llm_response.content.contains("Return ONLY this JSON")
        {
            return Err(anyhow!(
                "LLM returned prompt instructions as content — model may not support JSON mode. Raw response: {}",
                raw
            ));
        }

        // Sanitize the title
        let title = sanitize_title(&llm_response.title);

        let tasks = match self.generate_tasks(&llm_response.content, system_prompt).await {
            Ok(t) => t,
            Err(_) => vec![],
        };

        Ok(LlmResponse {
            title,
            content: llm_response.content,
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
