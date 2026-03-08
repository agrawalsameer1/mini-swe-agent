from pathlib import Path

class LegacyAgentMemory:
    def __init__(self, path: str):
        self.path = Path(path)

        if not self.path.exists():
            self.path.write_text(
                "=== ROLLING SUMMARY ===\n\n"
                "=== LATEST EXPLORATION ===\n"
            )

    def read(self) -> str:
        """Return the entire memory file as text."""
        return self.path.read_text()
    
    def reset(self):
        """Reset memory to the initial empty state."""
        self.path.write_text(
            "=== ROLLING SUMMARY ===\n\n"
            "=== LATEST EXPLORATION ===\n"
        )

    def add_exploration(self, new_text: str):
        """
        Move the previous LATEST EXPLORATION into ROLLING SUMMARY,
        then write the new exploration under LATEST EXPLORATION.
        """
        content = self.read()

        before_latest, latest_part = content.split("=== LATEST EXPLORATION ===", 1)
        summary_header, summary_text = before_latest.split("=== ROLLING SUMMARY ===", 1)

        previous_latest = latest_part.strip()

        # Append previous latest exploration to the rolling summary
        updated_summary = summary_text.strip()
        if previous_latest:
            updated_summary += "\n\n" + previous_latest

        new_content = (
            "=== ROLLING SUMMARY ===\n"
            + updated_summary.strip()
            + "\n\n=== LATEST EXPLORATION ===\n"
            + new_text.strip()
            + "\n"
        )

        self.path.write_text(new_content)

    def compress_summary(self, memory_summarizer):
        """
        Summarize the ROLLING SUMMARY section if memory gets too long.

        memory_summarizer: function taking (rolling_summary, latest_exploration)
        and returning a shorter rolling summary.
        """
        content = self.read()

        before_latest, latest_part = content.split("=== LATEST EXPLORATION ===", 1)
        _, summary_text = before_latest.split("=== ROLLING SUMMARY ===", 1)

        rolling_summary = summary_text.strip()
        latest_exploration = latest_part.strip()

        if not rolling_summary:
            return

        compressed = memory_summarizer(
            rolling_summary=rolling_summary,
            latest_exploration=latest_exploration,
        )

        new_content = (
            "=== ROLLING SUMMARY ===\n"
            + compressed.strip()
            + "\n\n=== LATEST EXPLORATION ===\n"
            + latest_exploration
            + "\n"
        )

        self.path.write_text(new_content)