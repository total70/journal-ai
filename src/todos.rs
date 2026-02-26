use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use ulid::Ulid;

use crate::providers::TaskItem;

#[derive(Debug, Serialize)]
struct TodoFrontmatter<'a> {
    id: &'a str,
    linked_note: &'a str,
    created: &'a str,
    updated: &'a str,
    status: &'a str, // pending | done | cancelled
    completed: Option<&'a str>,
    due: Option<&'a str>,
    priority: &'a str, // low | normal | high
    tags: Vec<String>,
}

fn default_journal_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join("Documents").join("journals"))
}

/// Reads ~/.config/file-journal/config.toml to find default_path.
pub fn read_file_journal_default_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let cfg_path = home
        .join(".config")
        .join("file-journal")
        .join("config.toml");

    if !cfg_path.exists() {
        return default_journal_path();
    }

    let content = fs::read_to_string(&cfg_path)
        .with_context(|| format!("Failed to read {}", cfg_path.display()))?;

    let value: toml::Value =
        toml::from_str(&content).context("Failed to parse file-journal config")?;
    if let Some(p) = value
        .get("default_path")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    {
        Ok(PathBuf::from(p))
    } else {
        default_journal_path()
    }
}

fn normalize_priority(p: &str) -> &str {
    match p {
        "low" | "normal" | "high" => p,
        _ => "normal",
    }
}

/// Save tasks as individual todo markdown files.
///
/// journal_root: path to journal root (e.g., ~/Documents/journals)
/// linked_note: relative path like 2026/02/25-220255-title.md
pub fn save_todos(
    journal_root: &Path,
    linked_note: &str,
    tasks: &[TaskItem],
) -> Result<Vec<PathBuf>> {
    if tasks.is_empty() {
        return Ok(vec![]);
    }

    let todos_dir = journal_root.join("todos");
    fs::create_dir_all(&todos_dir)
        .with_context(|| format!("Failed to create todos dir at {}", todos_dir.display()))?;

    let mut written = Vec::new();

    for task in tasks {
        let id = Ulid::new();
        let now = Utc::now();
        let now_iso = now.to_rfc3339_opts(SecondsFormat::Secs, true);
        let ts_prefix = now.format("%Y-%m-%d_%H%M%S").to_string();

        let filename = format!("{}_{}.md", ts_prefix, id);
        let path = todos_dir.join(filename);

        let due_opt = task.due.as_deref();
        let priority = normalize_priority(task.priority.as_str());

        let fm = TodoFrontmatter {
            id: &id.to_string(),
            linked_note,
            created: &now_iso,
            updated: &now_iso,
            status: "pending",
            completed: None,
            due: due_opt,
            priority,
            tags: vec![],
        };

        let yaml = serde_yaml::to_string(&fm).context("Failed to serialize todo frontmatter")?;
        let body = format!("---\n{}---\n\n{}\n", yaml, task.text.trim());

        fs::write(&path, body)
            .with_context(|| format!("Failed to write todo file {}", path.display()))?;

        written.push(path);
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::save_todos;
    use crate::providers::TaskItem;
    use tempfile::tempdir;

    #[test]
    fn test_save_todos_writes_files_with_frontmatter() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        let tasks = vec![TaskItem {
            text: "Bel Jan over Q2 planning".to_string(),
            priority: "normal".to_string(),
            due: Some("2026-02-27".to_string()),
        }];

        let written = save_todos(root, "2026/02/25-220255-test.md", &tasks).unwrap();
        assert_eq!(written.len(), 1);

        let content = std::fs::read_to_string(&written[0]).unwrap();
        assert!(content.contains("linked_note: 2026/02/25-220255-test.md"));
        assert!(content.contains("status: pending"));
        assert!(content.contains("priority: normal"));
        assert!(content.contains("due: 2026-02-27"));
        assert!(content.contains("Bel Jan over Q2 planning"));
    }
}
