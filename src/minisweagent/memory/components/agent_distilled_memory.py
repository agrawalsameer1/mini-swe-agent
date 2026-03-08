from pathlib import Path

class AgentDistilledMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.header = ""

        if not self.path.exists():
            self.path.write_text("")

    def read(self) -> str:
        """Return the entire memory file as text."""
        return self.path.read_text()

    def reset(self):
        """Reset memory to the initial empty state."""
        self.header = (
            "=== DISTILLED TASK MEMORY ===\n"
            "Persistent summary of key knowledge about the task gathered across all phases.\n\n"
        )

        self.path.write_text(self.header)

    def write(self, content: str):
        """Write content below the header."""
        self.path.write_text(self.header + content)