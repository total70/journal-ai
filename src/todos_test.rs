#[cfg(test)]
mod todos_tests {
    use super::super::todos;
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

        let written = todos::save_todos(root, "2026/02/25-220255-test.md", &tasks).unwrap();
        assert_eq!(written.len(), 1);

        let content = std::fs::read_to_string(&written[0]).unwrap();
        assert!(content.contains("linked_note: 2026/02/25-220255-test.md"));
        assert!(content.contains("status: pending"));
        assert!(content.contains("priority: normal"));
        assert!(content.contains("due: 2026-02-27"));
        assert!(content.contains("Bel Jan over Q2 planning"));
    }
}
