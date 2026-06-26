import os
import re
from pathlib import Path


MODEL_NAME = os.getenv("STOREPILOT_EMBEDDING_MODEL", "BAAI/bge-m3")
CACHE_ROOT = Path(os.getenv("STOREPILOT_AI_CACHE_ROOT", "ai-cache/categories"))
MODEL_CACHE_KEY = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME).strip("_").lower()

GUNPLA_CATEGORY_BONUS = float(os.getenv("STOREPILOT_GUNPLA_CATEGORY_BONUS", "0.3"))
BODY_KEYWORD_CATEGORY_BONUS = float(os.getenv("STOREPILOT_BODY_KEYWORD_CATEGORY_BONUS", "0.25"))
BODY_KEYWORD_EXACT_CATEGORY_BONUS = float(os.getenv("STOREPILOT_BODY_KEYWORD_EXACT_CATEGORY_BONUS", "0.15"))
BODY_KEYWORD_NON_MATCH_PENALTY = float(os.getenv("STOREPILOT_BODY_KEYWORD_NON_MATCH_PENALTY", "0.35"))

LLM_API_KEY = os.getenv("STOREPILOT_LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("STOREPILOT_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("STOREPILOT_LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = float(os.getenv("STOREPILOT_LLM_TIMEOUT_SECONDS", "90"))
LLM_BATCH_SIZE = int(os.getenv("STOREPILOT_LLM_BATCH_SIZE", "30"))
