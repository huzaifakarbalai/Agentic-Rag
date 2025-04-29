from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ConversationManager:
    def __init__(self, max_context_length: int = 5, include_answers: bool = True):
        """
        Initialize the conversation manager.

        Args:
            max_context_length: Maximum number of previous interactions to keep
            include_answers: Whether to include assistant's answers in the context
        """
        self.max_context_length = max_context_length
        self.include_answers = include_answers
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation history."""
        self.messages.append(
            Message(role=role, content=content, timestamp=datetime.now())
        )

        # Trim history if it exceeds max length
        if (
            len(self.messages) > self.max_context_length * 2
        ):  # *2 because each interaction has a Q&A pair
            self.messages = self.messages[-self.max_context_length * 2 :]

    def get_context(self) -> str:
        """Get the conversation context as a formatted string."""
        if not self.messages:
            return ""

        context_parts = []
        for msg in self.messages:
            if msg.role == "user" or (msg.role == "assistant" and self.include_answers):
                context_parts.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
