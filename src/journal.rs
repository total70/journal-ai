use anyhow::{anyhow, Context, Result};
use std::process::Command;

/// Check if file-journal is installed and available
pub fn check_file_journal() -> Result<()> {
    match Command::new("file-journal").arg("--help").output() {
        Ok(_) => Ok(()),
        Err(e) => Err(anyhow!(
            "file-journal not found in PATH. Please install it first: https://github.com/total70/file-journal\nError: {}",
            e
        )),
    }
}

/// Create a journal entry using file-journal
pub fn create_entry(title: &str, content: &str) -> Result<String> {
    // Ensure title ends with .md
    let title = if title.ends_with(".md") {
        title.to_string()
    } else {
        format!("{}.md", title)
    };

    let output = Command::new("file-journal")
        .arg("new")
        .arg(&title)
        .arg(content)
        .output()
        .context("Failed to execute file-journal. Is it installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("file-journal failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.trim().to_string())
}

/// Create a journal entry with dry-run (for testing)
pub fn create_entry_dry_run(title: &str, content: &str) -> Result<String> {
    let title = if title.ends_with(".md") {
        title.to_string()
    } else {
        format!("{}.md", title)
    };

    Ok(format!(
        "[DRY RUN] Would create:\n  Title: {}\n  Content: {}\n  Command: file-journal new '{}' '{}'",
        title, content, title, content
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_entry_dry_run() {
        let result = create_entry_dry_run("test-title", "Test content");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("test-title.md"));
        assert!(output.contains("Test content"));
    }

    #[test]
    fn test_title_with_md_extension() {
        let result = create_entry_dry_run("test.md", "Content");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("test.md"));
        assert!(!output.contains("test.md.md")); // Should not double the extension
    }
}
