import os
from typing import List, Tuple
from crewai import LLM
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from config.config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_RETRIES,
    CREW_LLM_MODEL,
    CREW_TEMPERATURE,
    CREW_MAX_TOKENS,
    ERROR_MESSAGES,
)


class LLMInterface:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")

        if not self.groq_api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(service="Groq"))
        if not self.gemini_api_key:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(service="Gemini"))

        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=None,
            max_retries=LLM_MAX_RETRIES,
        )

        self.crew_llm = LLM(
            model=CREW_LLM_MODEL,
            api_key=self.gemini_api_key,
            max_tokens=CREW_MAX_TOKENS,
            temperature=CREW_TEMPERATURE,
        )

    def check_local_knowledge(self, query: str, context: str) -> bool:
        """
        Check if the query can be answered from local knowledge.

        Args:
            query: User query
            context: Local context

        Returns:
            Boolean indicating if local knowledge is sufficient
        """
        prompt = """Role: Knowledge Verification Assistant
Task: Determine if the provided text contains sufficient information to answer the user's question.

Instructions:
1. Carefully analyze both the user's question and the provided text
2. Consider if the text contains:
   - Direct answers to the question
   - Sufficient context to infer a reliable answer
   - Relevant facts that could be combined to form an answer
3. Be strict in your evaluation - only answer "Yes" if you are confident the text contains enough information
4. Answer with a single word only: "Yes" or "No"

Examples:
Input: 
    Text: The capital of France is Paris, and it's known for the Eiffel Tower.
    Question: What is the capital of France?
Output: Yes

Input: 
    Text: The population of the United States is over 330 million.
    Question: What is the population of China?
Output: No

Input: 
    Text: The company was founded in 2010 and has offices in New York and London.
    Question: When was the company founded?
Output: Yes

Input: 
    Text: The company has offices in New York and London.
    Question: What is the company's revenue?
Output: No

Now evaluate:
Text: {text}
Question: {query}
"""
        try:
            formatted_prompt = prompt.format(text=context, query=query)
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip().lower() == "yes"
        except Exception as e:
            raise ValueError(ERROR_MESSAGES["llm_error"].format(error=str(e)))

    def generate_answer(self, context: str, query: str) -> str:
        """
        Generate an answer using the LLM.

        Args:
            context: Context for the query
            query: User query

        Returns:
            Generated answer
        """
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Use the provided context to answer the query accurately."
            ),
            SystemMessage(content=f"Context: {context}"),
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise ValueError(ERROR_MESSAGES["llm_error"].format(error=str(e)))
