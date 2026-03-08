import pytest
from memory import AgentMemory

def test_reset(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    text = memory.read()

    assert "=== ROLLING SUMMARY ===" in text
    assert "=== LATEST EXPLORATION ===" in text

def test_read_returns_string(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    text = memory.read()

    assert isinstance(text, str)
    assert len(text) > 0

def test_add_exploration_writes_latest(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    memory.add_exploration("Diagnosis A")

    text = memory.read()

    assert "Diagnosis A" in text
    assert "=== LATEST EXPLORATION ===" in text

def test_previous_latest_moves_to_summary(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    memory.add_exploration("First exploration")
    memory.add_exploration("Second exploration")

    text = memory.read()

    assert "First exploration" in text
    assert "Second exploration" in text

def test_latest_section_contains_only_latest(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    memory.add_exploration("Exploration A")
    memory.add_exploration("Exploration B")

    text = memory.read()

    latest_section = text.split("=== LATEST EXPLORATION ===")[1]

    assert "Exploration B" in latest_section

def test_multiple_updates_accumulate_summary(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)
    memory.reset()

    memory.add_exploration("A")
    memory.add_exploration("B")
    memory.add_exploration("C")

    text = memory.read()

    assert "A" in text
    assert "B" in text
    assert "C" in text

def test_file_created_if_missing(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)

    assert mem_file.exists()

def test_reset_clears_old_content(tmp_path):
    mem_file = tmp_path / "memory.txt"

    memory = AgentMemory(mem_file)

    memory.add_exploration("Something")
    memory.reset()

    text = memory.read()

    assert "Something" not in text