from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import Dict, Any
from config.config import ERROR_MESSAGES


class WebAgents:
    def __init__(self, llm):
        self.llm = llm
        self.search_tool = SerperDevTool()
        self.scrape_website = ScrapeWebsiteTool()

    def setup_agents(self) -> Crew:
        """
        Setup web search and scraping agents.

        Returns:
            Configured Crew instance
        """
        # Define the web search agent
        web_search_agent = Agent(
            role="Expert Web Search Agent",
            goal="Identify and retrieve relevant web data for user queries",
            backstory="An expert in identifying valuable web sources for the user's needs",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        # Define the web scraping agent
        web_scraper_agent = Agent(
            role="Expert Web Scraper Agent",
            goal="Extract and analyze content from specific web pages identified by the search agent",
            backstory="A highly skilled web scraper, capable of analyzing and summarizing website content accurately",
            allow_delegation=False,
            verbose=True,
            llm=self.llm,
        )

        # Define the web search task
        search_task = Task(
            description=(
                "Identify the most relevant web page or article for the topic: '{topic}'. "
                "Use all available tools to search for and provide a link to a web page "
                "that contains valuable information about the topic. Keep your response concise."
            ),
            expected_output=(
                "A concise summary of the most relevant web page or article for '{topic}', "
                "including the link to the source and key points from the content."
            ),
            tools=[self.search_tool],
            agent=web_search_agent,
        )

        # Define the web scraping task
        scraping_task = Task(
            description=(
                "Extract and analyze data from the given web page or website. Focus on the key sections "
                "that provide insights into the topic: '{topic}'. Use all available tools to retrieve the content, "
                "and summarize the key findings in a concise manner."
            ),
            expected_output=(
                "A detailed summary of the content from the given web page or website, highlighting the key insights "
                "and explaining their relevance to the topic: '{topic}'. Ensure clarity and conciseness."
            ),
            tools=[self.scrape_website],
            agent=web_scraper_agent,
        )

        # Create and return the crew
        return Crew(
            agents=[web_search_agent, web_scraper_agent],
            tasks=[search_task, scraping_task],
            verbose=1,
            memory=False,
        )

    def get_web_content(self, query: str) -> str:
        """
        Get content from web scraping.

        Args:
            query: Search query

        Returns:
            Retrieved web content
        """
        try:
            crew = self.setup_agents()
            result = crew.kickoff(inputs={"topic": query})
            return result.raw
        except Exception as e:
            raise ValueError(ERROR_MESSAGES["llm_error"].format(error=str(e)))
