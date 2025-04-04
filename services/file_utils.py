def load_markdown_file(file_path: str) -> str:
    """Load the entire markdown file as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
