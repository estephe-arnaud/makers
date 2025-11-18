"""
Agent Constants

Defines constants used for agent configuration and behavior.
"""

# LLM Temperature settings for different agent roles
AGENT_TEMPERATURE = 0.3  # Main agent: balanced creativity and accuracy
SUMMARY_LLM_TEMPERATURE = 0.1  # Summary agent: low temperature for factual consistency
DOCUMENT_ANALYST_TEMPERATURE = 0.2  # Document analyst: low temperature for analytical precision
DOCUMENT_SYNTHESIZER_TEMPERATURE = 0.4  # Document synthesizer: higher temperature for creative synthesis

