#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from my_stock_picker.crew import MyStockPicker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
target_sector = "Technology"  # Example: "Technology", "Finance", "Healthcare", etc.
max_pe_ratio = 40.0
min_market_cap = 500_000_000.0
preferred_news_keywords = ["innovation", "growth", "new contract"]
top_n_picks = 3

def run():
    """
    Run the crew.
    """

    # User-configurable parameters (can be adjusted as needed)
    inputs = {
        'target_sector': target_sector,
        'max_pe_ratio': max_pe_ratio,
        'min_market_cap': min_market_cap,
        'preferred_news_keywords': preferred_news_keywords,
        'top_n_picks': top_n_picks
    }

    
    try:
        result = MyStockPicker().crew().kickoff(inputs=inputs)
        # Print the result raw
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'target_sector': target_sector,
        'max_pe_ratio': max_pe_ratio,
        'min_market_cap': min_market_cap,
        'preferred_news_keywords': preferred_news_keywords,
        'top_n_picks': top_n_picks
    }
    try:
        MyStockPicker().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MyStockPicker().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'target_sector': target_sector,
        'max_pe_ratio': max_pe_ratio,
        'min_market_cap': min_market_cap,
        'preferred_news_keywords': preferred_news_keywords,
        'top_n_picks': top_n_picks
    }
    
    try:
        MyStockPicker().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
