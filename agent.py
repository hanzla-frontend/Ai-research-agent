"""
Builds a single-agent CrewAI "research crew".

The agent:
  - is a "Senior Research Analyst"
  - can search the web with the DuckDuckGo tool
  - is powered by Groq's `openai/gpt-oss-120b` model
  - writes a Markdown research report on whatever topic you give it
"""

import os
from crewai import Agent, Task, Crew, Process, LLM

from tools.search_tool import duckduckgo_search_tool


def build_crew(topic: str) -> Crew:
    """Create and return a ready-to-run Crew for the given research topic."""

    # ---- 1. The LLM (Groq, native support in CrewAI 1.15+, no LiteLLM needed) ----
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5,
    )

    # ---- 2. The Agent ----
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research the topic '{topic}' thoroughly using web search, "
            "then produce an accurate, well-organized, easy-to-read report."
        ),
        backstory=(
            "You are a meticulous research analyst at a respected think tank. "
            "You are known for never making things up, always verifying facts "
            "with real search results, and writing clear, well-structured reports "
            "that busy readers can skim in minutes."
        ),
        tools=[duckduckgo_search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # ---- 3. The Task ----
    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Steps:\n"
            "1. Use the DuckDuckGo Search Tool at least 2-3 times with different, "
            "specific queries to cover the topic from multiple angles "
            "(e.g. definition/background, recent developments, key facts or "
            "statistics, different viewpoints).\n"
            "2. Cross-check facts across the results you find.\n"
            "3. Write a clear, well-structured report in Markdown format.\n\n"
            "The report MUST include:\n"
            "- A short introduction (2-4 sentences)\n"
            "- Key findings organized under clear '##' subheadings\n"
            "- A brief conclusion / summary\n"
            "- A final '## Sources' section listing the links you actually used\n"
        ),
        expected_output=(
            "A complete, well-formatted Markdown research report on the topic, "
            "with headings, a conclusion, and a Sources section at the end."
        ),
        agent=researcher,
    )

    # ---- 4. The Crew (single agent -> sequential process) ----
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
