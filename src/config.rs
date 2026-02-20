use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_provider")]
    pub provider: String,
    
    #[serde(default)]
    pub ollama: OllamaConfig,
    
    #[serde(default)]
    pub openai: OpenAiConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            ollama: OllamaConfig::default(),
            openai: OpenAiConfig::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    
    #[serde(default = "default_ollama_model")]
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiConfig {
    #[serde(default = "default_openai_url")]
    pub base_url: String,
    
    #[serde(default = "default_openai_model")]
    pub model: String,
    
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            model: default_ollama_model(),
        }
    }
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: default_openai_url(),
            model: default_openai_model(),
            api_key: None,
        }
    }
}

fn default_provider() -> String {
    "ollama".to_string()
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "llama3.2".to_string()
}

fn default_openai_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_openai_model() -> String {
    "gpt-4o-mini".to_string()
}

impl Config {
    pub fn load(config_path: Option<PathBuf>) -> Result<Self> {
        // If explicit path provided, use that
        if let Some(path) = config_path {
            if path.exists() {
                let content = fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read config from {}", path.display()))?;
                let mut config: Config = toml::from_str(&content)
                    .with_context(|| "Failed to parse config TOML")?;
                config.load_api_keys();
                return Ok(config);
            }
        }

        // Try standard locations
        let config_paths = vec![
            Path::new(".journal-ai.toml").to_path_buf(),
            Self::default_config_path()?,
        ];

        for path in config_paths {
            if path.exists() {
                let content = fs::read_to_string(&path)
                    .with_context(|| format!("Failed to read config from {}", path.display()))?;
                let mut config: Config = toml::from_str(&content)
                    .with_context(|| "Failed to parse config TOML")?;
                config.load_api_keys();
                return Ok(config);
            }
        }

        // Return default config with env vars
        let mut config = Config::default();
        config.load_api_keys();
        Ok(config)
    }

    fn load_api_keys(&mut self) {
        // Load OpenAI API key from environment if not in config
        if self.openai.api_key.is_none() {
            if let Ok(key) = std::env::var("OPENAI_API_KEY") {
                self.openai.api_key = Some(key);
            }
        }
        
        // Also check ANTHROPIC_API_KEY for future use
        if let Ok(_key) = std::env::var("ANTHROPIC_API_KEY") {
            // Could be used for Anthropic provider in future
        }
    }

    pub fn default_config_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .context("Could not determine home directory")?;
        Ok(home.join(".config").join("journal-ai").join("config.toml"))
    }

    pub fn init_interactive() -> Result<Self> {
        println!("Welcome to journal-ai configuration!");
        println!();
        
        // Ask for default provider
        println!("Select default provider:");
        println!("1. Ollama (local, recommended for most users)");
        println!("2. OpenAI (cloud, requires API key)");
        
        let mut choice = String::new();
        std::io::stdin().read_line(&mut choice)?;
        
        let provider = match choice.trim() {
            "2" => "openai",
            _ => "ollama",
        };

        let mut config = Config::default();
        config.provider = provider.to_string();

        if provider == "openai" {
            println!("Enter your OpenAI API key (or set OPENAI_API_KEY env var):");
            let mut key = String::new();
            std::io::stdin().read_line(&mut key)?;
            config.openai.api_key = Some(key.trim().to_string());
        }

        // Save config
        let config_path = Self::default_config_path()?;
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Don't serialize api_key to file for security
        let config_to_save = Config {
            provider: config.provider.clone(),
            ollama: config.ollama.clone(),
            openai: OpenAiConfig {
                api_key: None, // Don't save API key to file
                ..config.openai.clone()
            },
        };

        let toml_string = toml::to_string_pretty(&config_to_save)?;
        fs::write(&config_path, toml_string)?;

        println!("Configuration saved to: {}", config_path.display());
        
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.ollama.model, "llama3.2");
        assert_eq!(config.openai.model, "gpt-4o-mini");
    }

    #[test]
    fn test_load_config_from_file() {
        let toml_content = r#"
provider = "openai"

[ollama]
base_url = "http://localhost:11434"
model = "mistral"

[openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4"
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();

        let config = Config::load(Some(temp_file.path().to_path_buf())).unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.ollama.model, "mistral");
        assert_eq!(config.openai.model, "gpt-4");
    }

    #[test]
    fn test_load_api_key_from_env() {
        std::env::set_var("OPENAI_API_KEY", "test-key-123");
        
        let mut config = Config::default();
        config.load_api_keys();
        
        assert_eq!(config.openai.api_key, Some("test-key-123".to_string()));
        
        std::env::remove_var("OPENAI_API_KEY");
    }
}
