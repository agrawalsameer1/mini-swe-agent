"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import traceback
from pathlib import Path

import time

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

import os
import uuid

import litellm
import copy

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.serialize import recursive_merge
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.phases.phase import Phase
from minisweagent.memory.components.agent_distilled_memory import AgentDistilledMemory
from minisweagent.memory.components.agent_handoff import AgentHandoff
from minisweagent.memory.components.agent_scratchpad import AgentScratchpad

from pathlib import Path
from pprint import pformat

MAXTOKENS = 4096

class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    distilled_memory_write_template: str
    exploration_handoff_write_template: str
    execution_report_write_template: str
    exploration_scratchpad_write_template: str
    execution_scratchpad_write_template: str

    phase_memory_update_template: str
    """Template for updating agent phase memory."""
    phase_recent_message_window: int
    """Number of recent messages the agent can see verbatim."""

    memory_write_template: str
    """Template for writing to agent memory."""
    memory_summarizer_template: str
    """Template for summarizing agent memory when it exceeds token limit."""

    exploration_system_template: str
    """System template used during exploration phase."""
    exploration_instance_template: str
    """User/task template used during exploration phase."""
    execution_system_template: str
    """System template used during execution phase."""
    execution_instance_template: str
    """User/task template used during execution phase."""
    
    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 2
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    steps: int = 0

class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.messages2: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

        self.phase = Phase.EXPLORATION

        self.task_id = uuid.uuid4().hex[:8]
        self.memory_dir = Path("current_agent_memory") / str(self.task_id)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.distilled_memory = AgentDistilledMemory(
            str(self.memory_dir / "distilled_memory.txt")
        )
        self.distilled_memory.reset()
        self.handoff = AgentHandoff(
            str(self.memory_dir / "handoff.txt")
        )
        self.handoff.reset(self.phase, initial=True)
        # self.scratchpad = AgentScratchpad("scratchpad.txt")
        # self.scratchpad.reset(self.phase)
        self.phase_start_idx = 2
        self.min_exploration_messages = 10

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str, **extra_vars) -> str:
        vars = self.get_template_vars()
        vars.update(extra_vars)
        return Template(template, undefined=StrictUndefined).render(**vars)

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def add_messages2(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages2.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        system_template, instance_template = self._get_phase_templates()
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(system_template)),
            self.model.format_message(role="user", content=self._render_template(instance_template)),
        )
        
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                print("Final token count: " + str(litellm.token_counter(model=self.model.config.model_name, messages=self.messages)))
                break

        return self.messages[-1].get("extra", {})
    
    def _refresh_phase_prompts(self) -> list[dict]:
        system_template, instance_template = self._get_phase_templates()

        prompt_messages: list[dict] = []

        system_msg = self.model.format_message(
            role="system",
            content=self._render_template(system_template),
        )
        prompt_messages.append(system_msg)

        instance_msg = self.model.format_message(
            role="user",
            content=self._render_template(instance_template),
        )
        prompt_messages.append(instance_msg)

        handoff_text = self.handoff.read().strip()
        if handoff_text:
            prompt_messages.append(
                self.model.format_message(role="user", content=handoff_text)
            )

        distilled_text = self.distilled_memory.read().strip()
        if distilled_text:
            prompt_messages.append(
                self.model.format_message(role="user", content=distilled_text)
            )

        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        if transcript:
            current_phase_messages = (
                "=== CURRENT PHASE CONTEXT ===\n"
                "These messages contain the transcript of the current phase, including your previous actions, command outputs, and reasoning.\n"
                "Use this information to decide whether to continue the phase or switch phases.\n\n"
                f"{transcript}"
            )

            prompt_messages.append(
                self.model.format_message(
                    role="user",
                    content=current_phase_messages,
                )
            )

        return prompt_messages

    def query(self) -> dict:
        """Query the model and return the model message."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        query_messages = self._refresh_phase_prompts()
        next_step_index = self.n_calls + 1

        logs_dir = Path("logs") / str(self.task_id)

        # Reset logs folder at the start of a run
        if self.n_calls == 0 and logs_dir.exists():
            import shutil
            shutil.rmtree(logs_dir)

        logs_dir.mkdir(parents=True, exist_ok=True)

        log_path = logs_dir / f"step_{next_step_index:03d}_{self.phase.value}.txt"

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"STEP: {next_step_index}\n")
            f.write(f"PHASE: {self.phase.value}\n")
            f.write("=" * 80 + "\n")
            f.write("FINAL BUILT PROMPT\n")
            f.write("=" * 80 + "\n\n")

            for i, msg in enumerate(query_messages, start=1):
                f.write(f"[MESSAGE {i}]\n")
                f.write(f"role: {msg.get('role', '')}\n")

                content = msg.get("content")
                if isinstance(content, str):
                    f.write("content:\n")
                    f.write(content)
                    f.write("\n")
                else:
                    f.write("content:\n")
                    f.write(pformat(content))
                    f.write("\n")

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    f.write("tool_calls:\n")
                    f.write(pformat(tool_calls))
                    f.write("\n")

                extra = msg.get("extra")
                if extra:
                    f.write("extra:\n")
                    f.write(pformat(extra))
                    f.write("\n")

                f.write("\n" + "-" * 80 + "\n\n")

        self.n_calls += 1
        print("[DEBUG] wrote prompt to log in query(), now calling self.model.query")
        message = self.model.query(query_messages)
        print("[DEBUG] self.model.query finished running")
        message.setdefault("extra", {})
        message["extra"]["phase"] = self.phase.value
        message["extra"]["step_index"] = self.n_calls
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)

        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODEL RESPONSE\n")
            f.write("=" * 80 + "\n\n")
            f.write(pformat(message))
            f.write("\n")

        return message
    
    def _get_phase_templates(self):
        if self.phase == Phase.EXPLORATION:
            return (
                self.config.exploration_system_template,
                self.config.exploration_instance_template,
            )
        if self.phase == Phase.EXECUTION:
            return (
                self.config.execution_system_template,
                self.config.execution_instance_template,
            )
        raise ValueError(f"Unknown phase: {self.phase}")

    def step(self) -> list[dict] | None:
        old_phase = self.phase
        print("\nNEW STEP, CURRENTLY AT START OF step()")
        print(f"[DEBUG] phase at start of step: {self.phase}")
        self.execute_actions(self.query())
        print(f"[DEBUG] phase after execute_actions: {self.phase}")

        if self.messages[-1].get("role") == "exit":
            return None

        # If phase changed during execute_actions, let the phase-switch logic
        # handle handoff/report generation, distilled memory update, and scratchpad reset.
        if self.phase != old_phase:
            return self.messages

        # if self.phase == Phase.EXPLORATION:
        #     scratchpad_text = self.create_exploration_scratchpad()
        # elif self.phase == Phase.EXECUTION:
        #     scratchpad_text = self.create_execution_scratchpad()
        # else:
        #     raise ValueError(f"Unknown phase: {self.phase}")

        # self.scratchpad.write(scratchpad_text)

        return self.messages

    def _can_leave_exploration(self) -> bool:
        exploration_messages = self.messages[self.phase_start_idx:]
        return len(exploration_messages) >= self.min_exploration_messages

    def _switch_phase(self, action: dict) -> dict:
        old_phase = self.phase
        new_phase = Phase(action["phase"])

        if old_phase == new_phase:
            return {
                "output": f"Already in phase {self.phase.value}.",
                "returncode": 0,
                "exception_info": "",
                "extra": {
                    "old_phase": old_phase.value,
                    "new_phase": new_phase.value,
                    "reason": action.get("reason", ""),
                },
            }
        
        # Update distilled task memory using the phase we are leaving.
        previous_distilled_memory = self.distilled_memory.read().strip()
        new_distilled_memory = self.create_distilled_memory(previous_distilled_memory)

        # Generate the outgoing handoff/report from the phase we are leaving.
        if old_phase == Phase.EXPLORATION and new_phase == Phase.EXECUTION:
            handoff_text = self.create_exploration_handoff(previous_distilled_memory)
        elif old_phase == Phase.EXECUTION and new_phase == Phase.EXPLORATION:
            handoff_text = self.create_execution_report(previous_distilled_memory)
        else:
            raise ValueError(f"Unsupported phase switch: {old_phase} -> {new_phase}")

        # Write memory artifacts before entering the new phase.
        self.handoff.reset(new_phase)
        self.handoff.write(handoff_text)

        self.distilled_memory.reset()
        self.distilled_memory.write(new_distilled_memory)

        # Enter the new phase with a fresh scratchpad.
        self.phase = new_phase
        self.phase_start_idx = len(self.messages) + 1

        # self.scratchpad.reset(self.phase)

        return {
            "output": (
                f"Switched phase from {old_phase.value} to {self.phase.value}.\n"
                f"Reason: {action.get('reason', '')}"
            ),
            "returncode": 0,
            "exception_info": "",
            "extra": {
                "old_phase": old_phase.value,
                "new_phase": self.phase.value,
                "reason": action.get("reason", ""),
            },
        }
    
    def execute_action(self, action: dict) -> dict:
        """Execute a single action."""
        tool = action.get("tool", "bash")

        if tool == "bash":
            return self.env.execute(action)

        if tool == "switch_phase":
            return self._switch_phase(action)

        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"Unknown tool: {tool}",
            "extra": {"tool": tool},
        }

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.execute_action(action) for action in message.get("extra", {}).get("actions", [])]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
    
    def _build_transcript(self, start_idx: int | None = None, end_idx: int | None = None) -> str:
        """
        Build a readable transcript of the selected messages containing only
        fields that are useful for reasoning, with step numbers.
        """
        msgs = self.messages[slice(start_idx, end_idx)]

        lines: list[str] = []
        step_num = 1

        for msg in msgs:
            role = msg.get("role", "")

            if role == "assistant":
                content = msg.get("content")
                tool_calls = msg.get("tool_calls") or []

                if content:
                    lines.append(f"STEP {step_num}")
                    lines.append("ASSISTANT THOUGHT:")
                    lines.append(content.strip())
                    lines.append("")
                    step_num += 1

                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    args = fn.get("arguments")

                    lines.append(f"STEP {step_num}")
                    lines.append("ASSISTANT TOOL CALL:")
                    if name:
                        lines.append(f"tool: {name}")
                    if args:
                        lines.append(f"arguments: {args}")
                    lines.append("")
                    step_num += 1

            elif role == "tool":
                lines.append(f"STEP {step_num}")
                lines.append("TOOL RESULT:")

                extra = msg.get("extra", {})
                returncode = extra.get("returncode")
                output = msg.get("content")
                exception = extra.get("exception_info")

                if returncode is not None:
                    lines.append(f"return_code: {returncode}")

                if output:
                    lines.append("output:")
                    lines.append(output.strip())

                if exception:
                    lines.append("exception:")
                    lines.append(exception.strip())

                lines.append("")
                step_num += 1

            elif role == "user":
                content = msg.get("content")
                if content:
                    lines.append(f"STEP {step_num}")
                    lines.append("USER MESSAGE:")
                    lines.append(content.strip())
                    lines.append("")
                    step_num += 1

        return "\n".join(lines).strip()

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

    def create_exploration_handoff(self, previous_memory: str) -> str:
        """
        Generate the exploration -> execution handoff from the current exploration phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = []

        return self._generate_memory_block(
            self.config.exploration_handoff_write_template,
            required_headers=required_headers,
            distilled_memory=previous_memory,
            transcript=transcript,
        )


    def create_execution_report(self, previous_memory: str) -> str:
        """
        Generate the execution -> exploration report from the current execution phase.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = []

        return self._generate_memory_block(
            self.config.execution_report_write_template,
            required_headers=required_headers,
            distilled_memory=previous_memory,
            transcript=transcript,
        )


    def create_distilled_memory(self, previous_memory: str) -> str:
        """
        Generate updated distilled task memory using the previous distilled memory
        plus the current phase transcript.
        """
        transcript = self._build_transcript(start_idx=self.phase_start_idx)

        required_headers = []

        return self._generate_memory_block(
            self.config.distilled_memory_write_template,
            required_headers=required_headers,
            existing_memory=previous_memory,
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
            "CURRENT HYPOTHESIS:",
            "CANDIDATE PATCH IDEA:",
            "VALIDATION IDEA:"
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
            "CURRENT PATCH STATE:",
            "NEXT EXECUTION ACTION:",
            "SWITCH CONDITION:",
        ]

        return self._generate_memory_block(
            self.config.execution_scratchpad_write_template,
            required_headers=required_headers,
            transcript=transcript,
        )