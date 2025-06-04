#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from my_financial_research.crew import MyFinancialResearch

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    # Set the company/ticker you want to research here
    inputs = {
        'company': 'Asian Alliance International PCL (BKK.AAI)',
    }
    try:
        result = MyFinancialResearch().crew().kickoff(inputs=inputs)
        ## print result.raw
        print(f"Crew executed successfully with result: {result.raw}")

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'company': 'Asian Alliance International PCL (BKK.AAI)',
    }
    try:
        MyFinancialResearch().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MyFinancialResearch().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'company': 'Asian Alliance International PCL (BKK.AAI)',
    }
    try:
        MyFinancialResearch().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
