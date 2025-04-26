from typing import Optional
from .data_loader import DataLoader
from .vector_store import VectorStore
from .llm_interface import LLMInterface
from .agents import WebAgents
from config.config import ERROR_MESSAGES


class RAGApplication:
    def __init__(self):
        self.data_loader = DataLoader()
        self.vector_store = VectorStore()
        self.llm_interface = LLMInterface()
        self.web_agents = WebAgents(self.llm_interface.crew_llm)

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

            # Generate final answer
            answer = self.llm_interface.generate_answer(context, query)
            return answer

        except Exception as e:
            raise ValueError(f"Error processing query: {str(e)}")


def main():
    # Initialize RAG application
    app = RAGApplication()
    app.initialize("data/tesla_q3.pdf")

    print("RAG Application initialized. Enter queries (Ctrl+C to exit):")

    while True:
        try:
            # Get query from user input
            query = input("\nEnter your query: ")

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
