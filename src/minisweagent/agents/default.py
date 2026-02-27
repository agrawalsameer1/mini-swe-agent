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

import litellm
import copy

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.serialize import recursive_merge
from minisweagent.models.litellm_model import LitellmModel

import vertexai
from vertexai.generative_models import GenerativeModel

MAXTOKENS = 4096

class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
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
        self.config.step_limit = 70
        self.messages: list[dict] = []
        self.messages2: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

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
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
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

    def query(self) -> dict:
        #Query the model and return model messages. Override to add hooks.
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        
        """
        while litellm.token_counter(model=self.model.config.model_name, messages=self.messages) > MAXTOKENS:
            self.messages.pop(2)
            while self.messages[2]["role"] != "assistant":
                self.messages.pop(2)
        """

        return message

    def step(self) -> list[dict]:
        self.execute_actions(self.query())
        if self.messages[-1].get("role") == "exit":
            return

        if litellm.token_counter(model=self.model.config.model_name, messages=self.messages) > MAXTOKENS:
            model2content = """Your role: You are a Summarizer Agent.\n
Your task: You will receive an Interaction History (a list of tool calls and outputs from a primary coding agent).\n
Synthesize this information into a brief, factual summary. The summary's purpose is to provide a condensed "memory" for the primary agent so it can decide its next action to solve its task.\n
The summary must capture:\n
1. The main objective.\n
2. Key actions taken and their results (e.g., files read, commands run).\n
3. Critical discoveries (e.g., location of a bug, relevant code snippets).\n
4. Failures or dead ends encountered.\n
5. The current state or focus of the investigation.\n
Output: Return ONLY the plain-text summary. Do not use JSON, Markdown, or any other formatting.\n
To assist you, here are the original system prompts given to the coding agent. NOTE that you do not have to include these in your summary, as these will always be passed in full to the coding agent:\n"""
            model2content += (self.messages[0]["content"] + "\n" + self.messages[1]["content"] + "\n\n\n\nINTERACTION HISTORY:\n")
            
            # Truncate history if too large
            history_str = str(self.messages[2:])
            last_assistant_idx = -1
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break
            messages_to_keep = []
            messages_to_summarize = []
        # Keep messages from the last assistant onwards (last step)
            if last_assistant_idx != -1:
                messages_to_keep = self.messages[last_assistant_idx:]
                messages_to_summarize = self.messages[2:last_assistant_idx]
            else:
            # No assistant message found, summarize everything except system/user
                messages_to_keep = []
                messages_to_summarize = self.messages[2:]
             
            vertexai.init(project=os.environ.get("VERTEX_PROJECT_ID"), location="us-central1")
            model2content += str(messages_to_summarize)
            # Exponential backoff retry logic
            max_retries = 5
            base_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Use stable model instead of experimental
                    summarizer = GenerativeModel("gemini-2.5-pro")
                    response = summarizer.generate_content(model2content)
                    summary_text = response.text
                    
                    print("\n\n\nSUMMARY:")
                    print(summary_text)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2 ** attempt)
                        print(f"Summarization attempt {attempt + 1} failed: {e}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        # Final attempt failed, use fallback
                        print(f"Summarization failed after {max_retries} attempts: {e}")
                        summary_text = "[Previous context truncated - summarization unavailable]"
        
            # Reset messages with summary
            self.messages = []
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
                self.model.format_message(role="assistant", content=summary_text),
            )

            for i in range(len(messages_to_keep)):
                self.messages.append(messages_to_keep[i])

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
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
