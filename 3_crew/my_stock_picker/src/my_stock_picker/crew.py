
# --- Stock Picker Crew Implementation ---
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field
from typing import Optional,List
# Example LLM import (adjust as needed for your environment)
from langchain_openai import OpenAI
# SerperDevTool for up-to-date web search/news
from crewai_tools import SerperDevTool
# PushNotificationTool for push notifications
from .tools.push_tool import PushNotificationTool

search_tool = SerperDevTool()


# --- User Configurable Parameters ---
target_sector = "Technology"  # Example: "Technology", "Finance", "Consumer Goods"
max_pe_ratio = 30.0
min_market_cap = 1_000_000_000.0
preferred_news_keywords = ["innovation", "growth", "new contract"]
top_n_picks = 1

# --- Pydantic Models for Structured Output ---
class StockCandidate(BaseModel):
    """Represents a stock candidate that passed the initial screening criteria."""
    ticker: str = Field(description="Stock ticker symbol.")
    company_name: str = Field(description="Name of the company.")
    reasoning: str = Field(description="Brief reason why this stock was selected based on screening criteria.")

class ScreenedStocks(BaseModel):
    """List of stocks that passed the initial screening criteria."""
    candidates: List[StockCandidate] = Field(description="List of stock candidates found during screening.")

class FundamentalAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol.")
    revenue_growth_yoy_percent: Optional[float] = Field(description="Year-over-year revenue growth percentage.")
    net_profit_margin_percent: Optional[float] = Field(description="Net profit margin percentage.")
    pe_ratio: Optional[float] = Field(description="Price-to-Earnings ratio.")
    debt_to_equity_ratio: Optional[float] = Field(description="Debt-to-Equity ratio.")
    business_model_summary: str = Field(description="Summary of the company's core business model.")
    competitive_advantages: List[str] = Field(description="List of key competitive advantages.")
    major_risks: List[str] = Field(description="List of major identifiable risks.")

class StockFundamentalsList(BaseModel):
    analyzed_stocks: List[FundamentalAnalysis] = Field(description="List of fundamental analyses for stocks.")

class SentimentAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol.")
    overall_sentiment: str = Field(description="Overall sentiment (e.g., Positive, Negative, Neutral).")
    key_news_items: List[str] = Field(description="List of key recent news items influencing sentiment.")
    analyst_consensus: Optional[str] = Field(description="Summary of analyst consensus if found.")

class StockSentimentsList(BaseModel):
    sentiment_analyses: List[SentimentAnalysis] = Field(description="List of sentiment analyses for stocks.")

class SelectedStockPick(BaseModel):
    ticker: str = Field(description="Stock ticker symbol.")
    company_name: str = Field(description="Name of the company.")
    investment_thesis: str = Field(description="Core reason for recommending this stock.")
    potential_catalysts: List[str] = Field(description="Potential events or factors that could drive the stock price up.")
    key_risks: List[str] = Field(description="Key risks associated with this investment.")

class FinalStockPicks(BaseModel):
    selected_picks: List[SelectedStockPick] = Field(description="The final list of selected stock picks.")
    disclaimer: str = Field(default="This analysis is for informational purposes only and does not constitute financial advice. Always conduct your own research or consult with a qualified financial advisor before making investment decisions.")



@CrewBase
class MyStockPicker():
    """MyStockPicker crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    


    @agent
    def market_screener(self) -> Agent:
        return Agent(
            config=self.agents_config['market_screener'],
            tools=[search_tool],
            verbose=True
        )

    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['fundamental_analyst'],
            verbose=True
        )


    @agent
    def news_sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['news_sentiment_analyst'],
            tools=[search_tool],
            verbose=True
        )
    
    @agent
    def stock_selection_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_selection_strategist'],
            tools=[PushNotificationTool()],
            verbose=True
        )

    @task
    def initial_stock_screening(self) -> Task:
        return Task(
            config=self.tasks_config['initial_stock_screening'],
            output_pydantic=ScreenedStocks,
            output_file='outputs/initial_stock_screening.json'
        )

    @task
    def fundamental_deep_dive(self) -> Task:
        return Task(
            config=self.tasks_config['fundamental_deep_dive'],
            output_pydantic=StockFundamentalsList,
            depends_on=[self.initial_stock_screening],
            output_file='outputs/fundamental_deep_dive.json'
        )

    @task
    def news_sentiment_review(self) -> Task:
        return Task(
            config=self.tasks_config['news_sentiment_review'],
            output_pydantic=StockSentimentsList,
            depends_on=[self.fundamental_deep_dive],
            output_file='outputs/news_sentiment_review.json'
        )

    @task
    def managed_stock_selection_reporting(self) -> Task:
        return Task(
            config=self.tasks_config['managed_stock_selection_reporting'],
            depends_on=[
                        self.initial_stock_screening,
                        self.fundamental_deep_dive,
                        self.news_sentiment_review
                    ],
                    output_file='outputs/managed_stock_selection_reporting.md'  # Markdown report as final output
                )

    @crew
    def crew(self) -> Crew:
        """Creates the Stock Picker Crew with hierarchical process and explicit manager agent"""
        # Instantiate the manager agent directly
        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True
        )
        return Crew(
            agents=self.agents,
            tasks=self.tasks, 
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
        )
