import json
from typing import Any, Dict, List, Optional

from mem0.memory.main import Memory


class MemoryTool:
    def __init__(self, memory: Memory):
        self.memory = memory

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add new information to memory.
        """
        return self.memory.add(content, user_id=user_id, agent_id=agent_id, run_id=run_id, metadata=metadata)

    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update existing memory.
        """
        final_metadata = metadata or {}
        if user_id:
            final_metadata["user_id"] = user_id
        if agent_id:
            final_metadata["agent_id"] = agent_id
        if run_id:
            final_metadata["run_id"] = run_id

        return self.memory.update(memory_id, content, metadata=final_metadata)

    def delete_memory(self, memory_id: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete memory.
        """
        return self.memory.delete(memory_id)

    def retrieve_memory(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant memories.
        """
        return self.memory.search(query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit)

    def summarize_context(self, messages: List[Dict[str, str]], context_window: int = 4000) -> str:
        """
        Summarize the conversation context.
        """
        from mem0.configs.prompts import SUMMARIZE_CONTEXT_PROMPT

        conversation_text = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in messages])

        prompt = SUMMARIZE_CONTEXT_PROMPT.format(conversation_text=conversation_text)

        response = self.memory.llm.generate_response(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        return response

    def filter_context(self, messages: List[Dict[str, str]], criteria: str) -> List[Dict[str, str]]:
        """
        Filter irrelevant information from the context.
        """
        from mem0.configs.prompts import FILTER_CONTEXT_PROMPT
        from mem0.memory.utils import extract_json

        conversation_text = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in messages])

        prompt = FILTER_CONTEXT_PROMPT.format(conversation_text=conversation_text, criteria=criteria)

        response = self.memory.llm.generate_response(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            # Try parsing JSON directly
            try:
                filtered_messages = json.loads(response)
            except json.JSONDecodeError:
                extracted = extract_json(response)
                filtered_messages = json.loads(extracted)

            if isinstance(filtered_messages, list):
                return filtered_messages
            # If wrapped in a key, try to find a list
            for key, value in filtered_messages.items():
                if isinstance(value, list):
                    return value

            return messages
        except Exception:
            # Fallback: return original if parsing fails
            return messages

    def reflect_on_memory(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Reflect on existing memories to generate high-level insights.
        """
        from mem0.configs.prompts import REFLECTION_PROMPT

        # 1. Retrieve relevant memories
        retrieved = self.memory.search(query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=10)

        if isinstance(retrieved, dict):
            results = retrieved.get("results", [])
        elif isinstance(retrieved, list):
            results = retrieved
        else:
            results = []

        if not results:
            return {"message": "No relevant memories found to reflect upon."}

        memories_text = "\n".join([f"- {mem.get('memory')}" for mem in results])

        # 2. Generate insight
        prompt = REFLECTION_PROMPT.format(query=query, memories=memories_text)

        insight = self.memory.llm.generate_response(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        # 3. Store the insight
        metadata = {"type": "reflection", "source_query": query}
        # Session IDs are handled by self.memory.add via arguments, but we can also put them in metadata for clarity/storage
        if user_id:
            metadata["user_id"] = user_id
        if agent_id:
            metadata["agent_id"] = agent_id
        if run_id:
            metadata["run_id"] = run_id

        added_memory = self.memory.add(insight, user_id=user_id, agent_id=agent_id, run_id=run_id, metadata=metadata)

        return {
            "insight": insight,
            "generated_memory": added_memory,
            "related_memories": results,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the JSON schema definitions for the memory tools.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "add_memory",
                    "description": "Add new information to memory. Useful for storing facts, user preferences, or important details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to store in memory."
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata to store with the memory."
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_memory",
                    "description": "Update existing memory. Useful when information changes or needs correction. Requires memory_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The ID of the memory to update."
                            },
                            "content": {
                                "type": "string",
                                "description": "The new content for the memory."
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Updated metadata."
                            }
                        },
                        "required": ["memory_id", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_memory",
                    "description": "Delete a specific memory by ID. Useful for removing obsolete or incorrect information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The ID of the memory to delete."
                            }
                        },
                        "required": ["memory_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memory",
                    "description": "Retrieve relevant memories based on a query. Useful for answering questions or providing context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for in memory."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of memories to retrieve (default 5)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "summarize_context",
                    "description": "Summarize the provided conversation messages to save context space.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "List of message objects (role, content) to summarize.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["messages"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filter_context",
                    "description": "Filter out irrelevant messages from the context based on criteria.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "List of message objects (role, content) to filter.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string"},
                                        "content": {"type": "string"}
                                    }
                                }
                            },
                            "criteria": {
                                "type": "string",
                                "description": "The criteria for filtering (e.g., 'remove greeting messages')."
                            }
                        },
                        "required": ["messages", "criteria"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reflect_on_memory",
                    "description": "Generate high-level insights from existing memories. Useful for synthesizing patterns or understanding user personality.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic or query to reflect upon."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
