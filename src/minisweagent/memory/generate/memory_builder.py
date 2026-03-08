from typing import Iterable

import litellm

from minisweagent.models.litellm_model import LitellmModel

class MemoryBuilder:
    def _build_transcript(self, start_idx: int | None = None, end_idx: int | None = None) -> str:
        """
        Build a compact transcript from self.messages[start_idx:end_idx].
        Includes message content and tool calls.
        """
        msgs = self.messages[slice(start_idx, end_idx)]
        transcript_parts: list[str] = []

        for msg in msgs:
            role = msg.get("role", "unknown")

            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                transcript_parts.append(f"{role.upper()}:\n{content.strip()}")

            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                tool_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", "")
                    tool_lines.append(f"- {name}({arguments})")
                transcript_parts.append(
                    f"{role.upper()} TOOL CALLS:\n" + "\n".join(tool_lines)
                )

        return "\n\n".join(transcript_parts)

    def _generate_memory_block(
        self,
        template: str,
        required_headers: Iterable[str],
        **template_kwargs,
    ) -> str:
        """
        Render a prompt template, call the model, and validate required headers.
        """
        prompt = self._render_template(template, **template_kwargs)
        messages = [
            self.model.format_message(role="system", content=prompt),
        ]

        response = litellm.completion(
            model=self.model.config.model_name,
            messages=messages,
            **self.model.config.model_kwargs,
        )

        text = response["choices"][0]["message"]["content"].strip()

        missing = [header for header in required_headers if header not in text]
        if missing:
            raise ValueError(f"Generated memory missing required headers: {missing}")

        return text

    def create_exploration_handoff(self) -> str:
        """
        Generate the exploration -> execution handoff from the current exploration phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = [
            "ISSUE SUMMARY:",
            "CURRENT HYPOTHESIS:",
            "TARGET FILES:",
            "PATCH PLAN:",
            "VALIDATION PLAN:",
            "RISKS / UNCERTAINTIES:",
            "SWITCH REASON:",
        ]

        return self._generate_memory_block(
            self.config.exploration_handoff_write_template,
            required_headers=required_headers,
            transcript=transcript,
        )

    def create_execution_report(self) -> str:
        """
        Generate the execution -> exploration report from the current execution phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = [
            "ATTEMPTED PATCH:",
            "FILES MODIFIED:",
            "VALIDATION RESULT:",
            "HYPOTHESIS STATUS:",
            "KEY OBSERVATIONS:",
            "NEXT EXPLORATION FOCUS:",
            "SWITCH REASON:",
        ]

        return self._generate_memory_block(
            self.config.execution_report_write_template,
            required_headers=required_headers,
            transcript=transcript,
        )

    def create_distilled_memory(self, previous_memory: str) -> str:
        """
        Generate updated distilled task memory using the previous distilled memory
        plus the current phase transcript.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = [
            "ISSUE CORE:",
            "CONFIRMED RELEVANT FILES:",
            "CONFIRMED IRRELEVANT PATHS:",
            "ACTIVE HYPOTHESES:",
            "INVALIDATED HYPOTHESES:",
            "ATTEMPT LEDGER:",
            "KNOWN REPRO / VALIDATION COMMANDS:",
            "IMPORTANT CONSTRAINTS:",
            "KEY CODE LOCATIONS:",
            "OPEN RISKS:",
        ]

        return self._generate_memory_block(
            self.config.distilled_memory_write_template,
            required_headers=required_headers,
            memory=previous_memory,
            transcript=transcript,
        )

    def create_exploration_scratchpad(self) -> str:
        """
        Generate the exploration scratchpad for the current exploration phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = [
            "FILES READ THIS PHASE:",
            "TESTS / REPROS INSPECTED:",
            "SEARCHES PERFORMED:",
            "CURRENT BEST HYPOTHESIS:",
            "MISSING INFORMATION:",
            "READY TO SWITCH:",
        ]

        return self._generate_memory_block(
            self.config.exploration_scratchpad_write_template,
            required_headers=required_headers,
            transcript=transcript,
        )

    def create_execution_scratchpad(self) -> str:
        """
        Generate the execution scratchpad for the current execution phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = [
            "FILES MODIFIED THIS PHASE:",
            "COMMANDS RUN:",
            "LATEST VALIDATION RESULT:",
            "CURRENT HYPOTHESIS STATUS:",
            "NEXT ACTION:",
            "READY TO SWITCH:",
        ]

        return self._generate_memory_block(
            self.config.execution_scratchpad_write_template,
            required_headers=required_headers,
            transcript=transcript,
        )