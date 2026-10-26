"""
Utility modules for SIKUKER application.
Contains prompt templates and formatting helpers.
"""

from app.utils.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_career_feedback_prompt,
)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "build_career_feedback_prompt",
]
