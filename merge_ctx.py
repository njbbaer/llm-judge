from pathlib import Path

intro_message = "Review the following codebase to understand the project."
script_path = Path(__file__).absolute()
output = ["# Codebase", "", intro_message, "", "---", ""]

for path in Path(".").rglob("*.py"):
    if path.absolute() == script_path:
        continue

    content = path.read_text()
    if not content.strip():
        content = "EMPTY FILE"
    output.extend([str(path), "", "```python", content, "```", "", "---", ""])

Path("ctx.md").write_text("\n".join(output))
