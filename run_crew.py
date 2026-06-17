import os
from dotenv import load_dotenv
import litellm

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Fix for Groq + CrewAI compatibility
litellm.cache = None
os.environ["LITELLM_CACHE"] = "false"

from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Research and summarize information about the given topic",
    backstory="You are an expert researcher who finds accurate information.",
    tools=[],
    llm="groq/llama-3.3-70b-versatile",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a clear structured response based on research findings",
    backstory="You are a skilled writer who explains things clearly.",
    tools=[],
    llm="groq/llama-3.3-70b-versatile",
    verbose=True
)

critic = Agent(
    role="Critic",
    goal="Review the written response for accuracy and completeness",
    backstory="You are a thorough reviewer who checks quality.",
    tools=[],
    llm="groq/llama-3.3-70b-versatile",
    verbose=True
)

task1 = Task(
    description="Research this topic: What is RAG and how is it used in AI?",
    expected_output="A detailed summary about RAG",
    agent=researcher
)

task2 = Task(
    description="Write a clear explanation about RAG based on the research",
    expected_output="A well written paragraph explaining RAG",
    agent=writer
)

task3 = Task(
    description="Review the explanation for accuracy and completeness",
    expected_output="A reviewed and improved final response about RAG",
    agent=critic
)

crew = Crew(
    agents=[researcher, writer, critic],
    tasks=[task1, task2, task3],
    verbose=True
)

result = crew.kickoff()
print("\n========== FINAL OUTPUT ==========")
print(result)