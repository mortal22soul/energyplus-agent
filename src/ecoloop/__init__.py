"""Eco-Loop autonomous building controller."""

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from the project root (parent of the src/ package)
# This ensures AZURE_OPENAI_API_KEY and other vars are available
# via os.getenv() in config.py, regardless of how the package is invoked.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=False
)

from .schemas import BuildingState, ControlAction, ZoneSetpoints

__all__ = ["BuildingState", "ControlAction", "ZoneSetpoints"]
