import pytest
from unittest.mock import MagicMock, patch
from mem0.memory.tools import MemoryTool

@pytest.fixture
def mock_memory():
    memory = MagicMock()
    # Mock LLM
    memory.llm = MagicMock()
    return memory

@pytest.fixture
def memory_tool(mock_memory):
    return MemoryTool(mock_memory)

def test_add_memory(memory_tool, mock_memory):
    content = "Test content"
    metadata = {"key": "value"}
    user_id = "user1"

    mock_memory.add.return_value = {"id": "mem1"}

    result = memory_tool.add_memory(content, metadata, user_id=user_id)

    mock_memory.add.assert_called_once_with(
        content, user_id=user_id, agent_id=None, run_id=None, metadata=metadata
    )
    assert result == {"id": "mem1"}

def test_update_memory(memory_tool, mock_memory):
    memory_id = "mem1"
    content = "Updated content"
    metadata = {"key": "new_value"}
    user_id = "user1"

    mock_memory.update.return_value = {"message": "success"}

    result = memory_tool.update_memory(memory_id, content, metadata=metadata, user_id=user_id)

    mock_memory.update.assert_called_once_with(
        memory_id, content, metadata={"key": "new_value", "user_id": "user1"}
    )
    assert result == {"message": "success"}

def test_delete_memory(memory_tool, mock_memory):
    memory_id = "mem1"

    mock_memory.delete.return_value = {"message": "deleted"}

    result = memory_tool.delete_memory(memory_id)

    mock_memory.delete.assert_called_once_with(memory_id)
    assert result == {"message": "deleted"}

def test_retrieve_memory(memory_tool, mock_memory):
    query = "test query"
    user_id = "user1"

    mock_memory.search.return_value = {"results": [{"id": "mem1", "memory": "content"}]}

    result = memory_tool.retrieve_memory(query, user_id=user_id)

    mock_memory.search.assert_called_once_with(
        query, user_id=user_id, agent_id=None, run_id=None, limit=5
    )
    assert result == {"results": [{"id": "mem1", "memory": "content"}]}

def test_summarize_context(memory_tool, mock_memory):
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    mock_memory.llm.generate_response.return_value = "Summary: User greeted assistant."

    result = memory_tool.summarize_context(messages)

    assert "Summary: User greeted assistant." in result
    mock_memory.llm.generate_response.assert_called_once()
    call_args = mock_memory.llm.generate_response.call_args
    assert "messages" in call_args.kwargs
    assert "summarization" in call_args.kwargs["messages"][1]["content"]

def test_filter_context(memory_tool, mock_memory):
    messages = [{"role": "user", "content": "Hello"}, {"role": "user", "content": "Task info"}]
    criteria = "remove greetings"

    # Mock LLM to return JSON string
    mock_response = '{"filtered": [{"role": "user", "content": "Task info"}]}'
    mock_memory.llm.generate_response.return_value = mock_response

    result = memory_tool.filter_context(messages, criteria)

    # The tool implementation tries to find a list in the response values
    assert len(result) == 1
    assert result[0]["content"] == "Task info"

def test_reflect_on_memory(memory_tool, mock_memory):
    query = "user personality"
    user_id = "user1"

    # 1. Search returns memories
    mock_memory.search.return_value = {"results": [{"id": "1", "memory": "User likes coffee"}]}

    # 2. LLM returns insight
    mock_memory.llm.generate_response.return_value = "User is a coffee lover."

    # 3. Add stores insight
    mock_memory.add.return_value = {"id": "new_mem"}

    result = memory_tool.reflect_on_memory(query, user_id=user_id)

    mock_memory.search.assert_called_once()
    mock_memory.llm.generate_response.assert_called_once()
    mock_memory.add.assert_called_once()

    assert result["insight"] == "User is a coffee lover."
    assert result["generated_memory"] == {"id": "new_mem"}

def test_get_tool_definitions(memory_tool):
    defs = memory_tool.get_tool_definitions()
    assert isinstance(defs, list)
    assert len(defs) == 7
    tool_names = [d["function"]["name"] for d in defs]
    assert "reflect_on_memory" in tool_names
    assert "add_memory" in tool_names
