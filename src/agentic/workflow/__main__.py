"""
Main entry point for testing the workflow.
"""

import asyncio
from config.logging_config import setup_logging
from src.agentic.workflow.runner import run_workflow


async def main():
    """Main function to run a test of the research workflow."""
    test_query = "What are the latest developments in explainable AI (XAI)?"
    result = await run_workflow(test_query)

    print("\n--- WORKFLOW EXECUTION FINISHED ---")
    if "error" in result:
        print(f"\nTest failed: {result['error']}")
    else:
        print("\nTest completed successfully!")
        print(f"Thread ID: {result['thread_id']}")
        print("\n--- FINAL OUTPUT ---")
        print(result.get("output", "Not available."))
    print("---------------------------------")


if __name__ == "__main__":
    setup_logging(level="INFO")
    asyncio.run(main())

