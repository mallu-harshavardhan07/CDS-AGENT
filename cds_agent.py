"""
=============================================================================
Module: cds_agent.py
Description: Companion alias for 04_cds_agent.py to enable standard Python
             identifier imports (e.g., 'from cds_agent import CDSAgent').
Author: SAP S/4HANA Cloud & AI Integration Engineering
=============================================================================
"""

import importlib

# Import all public symbols from 04_cds_agent
_agent_module = importlib.import_module("04_cds_agent")

CDSAgent = getattr(_agent_module, "CDSAgent")
CDSCleanCoreAgent = getattr(_agent_module, "CDSCleanCoreAgent", CDSAgent)
CDSGenerationResult = getattr(_agent_module, "CDSGenerationResult")
CleanCoreMockGenerator = getattr(_agent_module, "CleanCoreMockGenerator")
LiveLLMCaller = getattr(_agent_module, "LiveLLMCaller")

__all__ = [
    "CDSAgent",
    "CDSCleanCoreAgent",
    "CDSGenerationResult",
    "CleanCoreMockGenerator",
    "LiveLLMCaller"
]
