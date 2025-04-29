from typing import Optional
from .data_loader import DataLoader
from .vector_store import VectorStore
from .llm_interface import LLMInterface
from .agents import WebAgents
from .conversation_manager import ConversationManager
from config.config import ERROR_MESSAGES


class RAGApplication:
    def __init__(self, max_context_length: int = 5, include_answers: bool = True):
        self.data_loader = DataLoader()
        self.vector_store = VectorStore()
        self.llm_interface = LLMInterface()
        self.web_agents = WebAgents(self.llm_interface.crew_llm)
        self.conversation_manager = ConversationManager(
            max_context_length, include_answers
        )

    def initialize(self, document_path: str) -> None:
        """
        Initialize the RAG application with a document.

        Args:
            document_path: Path to the document to load

        Raises:
            FileNotFoundError: If the document doesn't exist
            ValueError: If there's an error loading or processing the document
        """
        try:
            documents = self.data_loader.load_documents(document_path)
            self.vector_store.create_vector_store(documents)
        except Exception as e:
            raise ValueError(f"Error initializing RAG application: {str(e)}")

    def process_query(self, query: str) -> str:
        """
        Process a user query and return an answer.

        Args:
            query: User query

        Returns:
            Generated answer

        Raises:
            ValueError: If there's an error processing the query
        """
        try:
            # Add user query to conversation history
            self.conversation_manager.add_message("user", query)

            # Get conversation context
            conversation_context = self.conversation_manager.get_context()

            # Get initial context for routing
            local_context = self.vector_store.get_context(query)

            # Check if we can answer from local knowledge
            can_answer_locally = self.llm_interface.check_local_knowledge(
                query, local_context
            )

            print(f"Can answer locally: {can_answer_locally}")

            # Get context either from local DB or web
            if can_answer_locally:
                context = self.vector_store.get_context(query)
            else:
                context = self.web_agents.get_web_content(query)

            # Combine conversation context with retrieved context
            full_context = f"Previous conversation:\n{conversation_context}\n\nRetrieved information:\n{context}"

            # Generate final answer
            answer = self.llm_interface.generate_answer(full_context, query)

            # Add assistant's answer to conversation history
            self.conversation_manager.add_message("assistant", answer)

            return answer

        except Exception as e:
            raise ValueError(f"Error processing query: {str(e)}")

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_manager.clear()


def main():
    # Initialize RAG application
    app = RAGApplication(max_context_length=5, include_answers=True)
    app.initialize("data/tesla_q3.pdf")

    print("RAG Application initialized. Enter queries (Ctrl+C to exit):")
    print("Type 'clear' to clear conversation history")

    while True:
        try:
            # Get query from user input
            query = input("\nEnter your query: ")

            if query.lower() == "clear":
                app.clear_conversation()
                print("Conversation history cleared.")
                continue

            # Process query and print answer
            answer = app.process_query(query)
            print(f"\nAnswer: {answer}")

        except KeyboardInterrupt:
            print("\nExiting application...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
