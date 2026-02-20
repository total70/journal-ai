use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::io::{self, Read};

mod config;
mod journal;
mod providers;

use config::Config;
use providers::{ollama::OllamaProvider, openai::OpenAiProvider, LlmProvider};

#[derive(Parser)]
#[command(name = "journal-ai")]
#[command(about = "AI-powered journal entry creation")]
#[command(version = "0.1.0")]
struct Cli {
    /// The note content (optional, can also use stdin)
    content: Option<String>,

    /// Provider to use (ollama, openai)
    #[arg(short, long)]
    provider: Option<String>,

    /// Model to use
    #[arg(short, long)]
    model: Option<String>,

    /// Path to config file
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,

    /// Dry run - don't actually create the entry
    #[arg(long)]
    dry_run: bool,

    /// Show what would be created without saving
    #[arg(long)]
    preview: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize configuration
    Init,
    /// Check if everything is set up correctly
    Doctor,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands
    match cli.command {
        Some(Commands::Init) => {
            Config::init_interactive()?;
            return Ok(());
        }
        Some(Commands::Doctor) => {
            run_doctor().await?;
            return Ok(());
        }
        None => {}
    }

    // Load configuration
    let mut config = Config::load(cli.config)?;

    // Override provider if specified
    if let Some(provider) = cli.provider {
        config.provider = provider;
    }

    // Override model if specified
    if let Some(model) = cli.model {
        match config.provider.as_str() {
            "ollama" => config.ollama.model = model,
            "openai" => config.openai.model = model,
            _ => eprintln!("Warning: Unknown provider, model override ignored"),
        }
    }

    // Get input content
    let content = match cli.content {
        Some(c) => c,
        None => {
            // Try to read from stdin
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            if buffer.trim().is_empty() {
                return Err(anyhow::anyhow!(
                    "No content provided. Use positional argument or pipe content via stdin.\n\
                     Example: journal-ai 'My note here'\n\
                     Or: echo 'My note' | journal-ai"
                ));
            }
            buffer.trim().to_string()
        }
    };

    // Check if file-journal is available
    journal::check_file_journal()
        .context("file-journal check failed")?;

    // Create provider with fallback logic
    let provider: Box<dyn LlmProvider> = match config.provider.as_str() {
        "ollama" => {
            let provider = OllamaProvider::new(config.ollama.clone());
            if !provider.is_available() {
                eprintln!("Warning: Ollama does not appear to be available at {}", config.ollama.base_url);
                eprintln!("Make sure Ollama is running: ollama serve");
                eprintln!("Attempting anyway...");
            }
            Box::new(provider)
        }
        "openai" => {
            let provider = OpenAiProvider::new(config.openai.clone())?;
            if !provider.is_available() {
                return Err(anyhow::anyhow!(
                    "OpenAI provider not available. Make sure OPENAI_API_KEY is set."
                ));
            }
            Box::new(provider)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown provider: {}. Use 'ollama' or 'openai'",
                config.provider
            ));
        }
    };

    // Generate structured entry
    println!("Generating journal entry using {}...", config.provider);
    
    let response = provider.generate(&content, None).await
        .with_context(|| format!("Failed to generate entry using {}", config.provider))?;

    // Preview mode - just show what would be created
    if cli.preview || cli.dry_run {
        println!("\n=== Preview ===");
        println!("Title: {}", response.title);
        println!("Content: {}", response.content);
        if !response.tags.is_empty() {
            println!("Tags: {}", response.tags.join(", "));
        }
        
        if cli.dry_run {
            let result = journal::create_entry_dry_run(&response.title, &response.content)?;
            println!("\n{}", result);
        }
        
        return Ok(());
    }

    // Create the actual entry
    println!("Saving entry: {}", response.title);
    
    let result = journal::create_entry(&response.title, &response.content)?;
    println!("{}", result);

    Ok(())
}

async fn run_doctor() -> Result<()> {
    println!("Running doctor check...\n");

    // Check config
    match Config::load(None) {
        Ok(config) => {
            println!("✓ Configuration loaded");
            println!("  Provider: {}", config.provider);
            match config.provider.as_str() {
                "ollama" => {
                    println!("  Model: {}", config.ollama.model);
                    println!("  URL: {}", config.ollama.base_url);
                }
                "openai" => {
                    println!("  Model: {}", config.openai.model);
                    println!("  API Key: {}", 
                        if config.openai.api_key.is_some() { "Set" } else { "Not set" }
                    );
                }
                _ => {}
            }
        }
        Err(e) => {
            println!("✗ Configuration issue: {}", e);
            println!("  Run 'journal-ai init' to set up configuration");
        }
    }

    // Check file-journal
    match journal::check_file_journal() {
        Ok(_) => println!("✓ file-journal is installed"),
        Err(e) => println!("✗ file-journal not found: {}", e),
    }

    // Check Ollama if configured as provider
    if let Ok(config) = Config::load(None) {
        if config.provider == "ollama" {
            let client = reqwest::Client::new();
            match client.get(format!("{}/api/tags", config.ollama.base_url)).send().await {
                Ok(resp) if resp.status().is_success() => {
                    println!("✓ Ollama is running at {}", config.ollama.base_url);
                }
                _ => {
                    println!("✗ Ollama not reachable at {}", config.ollama.base_url);
                    println!("  Make sure Ollama is running: ollama serve");
                }
            }
        }
    }

    println!("\nDoctor check complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::parse_from(["journal-ai", "test content"]);
        assert_eq!(cli.content, Some("test content".to_string()));
    }

    #[test]
    fn test_cli_with_provider() {
        let cli = Cli::parse_from(["journal-ai", "-p", "openai", "test"]);
        assert_eq!(cli.provider, Some("openai".to_string()));
    }
}
