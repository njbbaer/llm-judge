from pathlib import Path

intro_message = "Review the following codebase to understand the project."
script_path = Path(__file__).absolute()
output = ["# Codebase", "", intro_message, "", "---", ""]
for path in Path(".").rglob("*.py"):
    if path.absolute() == script_path:
        continue
    output.extend(
        [str(path), "", "```python", path.read_text(), "```", "", "---", "", ""]
    )

Path("ctx.md").write_text("\n".join(output))
