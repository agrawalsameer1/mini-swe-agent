from pathlib import Path

from minisweagent.phases.phase import Phase

class AgentScratchpad:
    def __init__(self, path: str):
        self.path = Path(path)
        self.header = ""

        if not self.path.exists():
            self.path.write_text("")

    def read(self) -> str:
        """Return the entire memory file as text."""
        return self.path.read_text()

    def reset(self, phase: Phase):
        """Reset memory to the initial empty state."""
        if phase == Phase.EXPLORATION:
            self.header = (
                "=== EXPLORATION SCRATCHPAD ===\n"
                "Temporary working notes for the current exploration phase. Reset on phase switch.\n\n"
            )
        elif phase == Phase.EXECUTION:
            self.header = (
                "=== EXECUTION SCRATCHPAD ===\n"
                "Temporary working notes for the current execution phase. Reset on phase switch.\n\n"
            )
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self.path.write_text(self.header)

    def write(self, content: str):
        """Write content below the header."""
        self.path.write_text(self.header + content)