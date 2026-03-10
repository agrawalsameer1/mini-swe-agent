from pathlib import Path

from minisweagent.phases.phase import Phase

class AgentHandoff:
    def __init__(self, path: str):
        self.path = Path(path)
        self.header = ""

        if not self.path.exists():
            self.path.write_text("")

    def read(self) -> str:
        return self.path.read_text()

    def reset(self, phase: Phase, initial: bool = False):
        """Reset the handoff memory for the new phase."""
        if phase == Phase.EXPLORATION:
            if initial:
                self.header = (
                    "=== CURRENT OBJECTIVE ===\n"
                    "Initial exploration of the issue. Identify the relevant code "
                    "implementing the behavior described in the task and prepare "
                    "a patch plan.\n\n"
                )
            else:
                self.header = (
                    "=== CURRENT OBJECTIVE ===\n"
                    "Continue exploration based on the previous execution report.\n\n"
                )

        elif phase == Phase.EXECUTION:
            self.header = (
                "=== CURRENT OBJECTIVE ===\n"
                "Execute the patch plan identified during exploration.\n\n"
            )
        
        elif phase == Phase.TESTING:
            self.header = (
                "=== CURRENT OBJECTIVE ===\n"
                "Evaluate the fix using tests.\n\n"
            )

        else:
            raise ValueError(f"Unknown phase: {phase}")

        self.path.write_text(self.header)

    def write(self, content: str):
        """Write content below the header."""
        self.path.write_text(self.header + content)