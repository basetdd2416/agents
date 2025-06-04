
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# Tool import for financial data gatherer
import os
from crewai_tools import SerperDevTool

# Make sure to set your SERPER_API_KEY in your environment before running!
# os.environ["SERPER_API_KEY"] = "YOUR_API_KEY"
search_tool = SerperDevTool()

@CrewBase
class MyFinancialResearch():
    """MyFinancialResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools


    @agent
    def financial_data_gatherer(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_data_gatherer'], # type: ignore[index]
            tools=[search_tool],
            verbose=True
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'], # type: ignore[index]
            verbose=True
        )

    @agent
    def report_generation(self) -> Agent:
        return Agent(
            config=self.agents_config['report_generation'], # type: ignore[index]
            verbose=True
        )

    @task
    def data_collection(self) -> Task:
        return Task(
            config=self.tasks_config['data_collection'], # type: ignore[index]
        )

    @task
    def in_depth_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['in_depth_analysis'], # type: ignore[index]
        )

    @task
    def report_compilation(self) -> Task:
        return Task(
            config=self.tasks_config['report_compilation'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyFinancialResearch crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
