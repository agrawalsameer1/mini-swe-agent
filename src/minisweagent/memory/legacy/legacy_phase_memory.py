from pathlib import Path

class LegacyAgentPhaseMemory:
    def __init__(self, path: str):
        self.path = Path(path)

        if not self.path.exists():
            self.path.write_text(
                "=== CURRENT PHASE ===\n"
            )

    def read(self) -> str:
        """Return the entire memory file as text."""
        return self.path.read_text()
    
    def reset(self):
        """Reset memory to the initial empty state."""
        self.path.write_text(
            "=== CURRENT PHASE ===\n"
        )

    def write(self, latest_step: str, updater_fn) -> str:
        """
        Update the phase memory using the previous memory plus the latest step.

        Args:
            latest_step: Text describing the most recent step bundle
                (e.g. assistant reasoning, tool call, tool result).
            updater_fn: Callable taking (previous_memory: str, latest_step: str)
                and returning updated memory text.

        Returns:
            The updated memory text.
        """
        previous_memory = self.read()

        new_memory = updater_fn(previous_memory, latest_step).strip()

        self.path.write_text(new_memory)
        return new_memory