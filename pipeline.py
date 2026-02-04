# nlp/pipeline.py
import os
import re
import json
import time
import random
import logging
from typing import Tuple, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "nlp.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Optional OpenAI
OpenAI = None
_openai_available = False
try:
    from openai import OpenAI as _OpenAI
    OpenAI = _OpenAI
    _openai_available = True
except Exception:
    logger.info("OpenAI SDK not available")

# Optional Ollama
ollama = None
_ollama_available = False
try:
    import ollama
    _ollama_available = True
except Exception:
    logger.info("Ollama not available")


class NLPPipeline:
    def __init__(
        self,
        csv_file: Optional[str] = None,
        intents_file: Optional[str] = None,
        similarity_threshold: float = 0.1,
        openai_model: str = "gpt-4o-mini",
        ollama_model: Optional[str] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.openai_model = openai_model
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "medgemma")
        self.base_dir = BASE_DIR

        csv_file = csv_file or os.path.join(self.base_dir, "data.csv")
        intents_file = intents_file or os.path.join(self.base_dir, "intents.json")

        # Load CSV
        if os.path.exists(csv_file):
            try:
                self.data = pd.read_csv(csv_file).dropna(subset=["statement", "status"])
                self.data["statement"] = self.data["statement"].astype(str)
                self.data["status"] = self.data["status"].astype(str)
            except Exception:
                self.data = pd.DataFrame(columns=["statement", "status"])
        else:
            self.data = pd.DataFrame(columns=["statement", "status"])

        # Load intents JSON
        try:
            with open(intents_file, "r", encoding="utf-8") as f:
                self.intents_json = json.load(f)
        except Exception:
            self.intents_json = {"intents": []}

        # Build intent → responses
        self.intents = {
            intent["tag"].lower(): intent.get("responses", [])
            for intent in self.intents_json.get("intents", [])
            if intent.get("tag")
        }

        # Build training patterns
        self.patterns_norm = []
        self.tags = []

        for _, row in self.data.iterrows():
            self.patterns_norm.append(self.normalize_text(row["statement"]))
            self.tags.append(row["status"].lower())

        for intent in self.intents_json.get("intents", []):
            tag = intent.get("tag", "").lower()
            for p in intent.get("patterns", []):
                self.patterns_norm.append(self.normalize_text(p))
                self.tags.append(tag)

        # TF-IDF
        if self.patterns_norm:
            self.vectorizer = TfidfVectorizer().fit(self.patterns_norm)
            self.patterns_vec = self.vectorizer.transform(self.patterns_norm)
        else:
            self.vectorizer = None
            self.patterns_vec = None

        # Cache
        self.cache_file = os.path.join(self.base_dir, "llm_cache.json")
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

        # OpenAI client
        self.openai_client = None
        if _openai_available and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logger.info("Pipeline loaded. Intents: %s", list(self.intents.keys()))

    # ---------------- utilities ----------------

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text)

    def _read_cache(self) -> dict:
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_cache(self, cache: dict):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    # ---------------- LLM fallback ----------------

    def llm_fallback(self, prompt: str) -> str:
        cache = self._read_cache()
        if prompt in cache:
            return cache[prompt]["response"]

        # OpenAI
        if self.openai_client:
            try:
                r = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                content = r.choices[0].message.content.strip()
                cache[prompt] = {"response": content, "ts": time.time()}
                self._write_cache(cache)
                return content
            except Exception:
                pass

        # Ollama
        if _ollama_available:
            try:
                r = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = r["message"]["content"]
                cache[prompt] = {"response": content, "ts": time.time()}
                self._write_cache(cache)
                return content
            except Exception:
                pass

        return random.choice(self.intents.get("fallback", ["I’m here with you 💙"]))

    # ---------------- main API ----------------

    def get_response(self, text: str) -> Tuple[str, Dict]:
        if not text or not text.strip():
            return "Hello 🙂", {"intent": "fallback", "confidence": 0.0}

        text_clean = self.normalize_text(text)

        # HARD STOP greetings
        if text_clean in {"hi", "hello", "hey", "hiya", "yo"}:
            return (
                random.choice(self.intents.get("greeting", ["Hello 🙂"])),
                {"intent": "greeting", "confidence": 1.0},
            )

        # Exact / substring match
        for intent in self.intents_json.get("intents", []):
            tag = intent.get("tag", "").lower()
            for p in intent.get("patterns", []):
                p_norm = self.normalize_text(p)
                if text_clean == p_norm or p_norm in text_clean or text_clean in p_norm:
                    responses = self.intents.get(tag)
                    if responses:
                        return random.choice(responses), {"intent": tag, "confidence": 1.0}

        # Skip TF-IDF for short text
        if len(text_clean.split()) <= 2:
            return self.llm_fallback(text), {"intent": "fallback", "confidence": 0.0}

        # TF-IDF similarity
        try:
            vec = self.vectorizer.transform([text_clean])
            sim = cosine_similarity(vec, self.patterns_vec)
            idx = sim.argmax()
            score = float(sim[0][idx])
        except Exception:
            return self.llm_fallback(text), {"intent": "fallback", "confidence": 0.0}

        if score < self.similarity_threshold:
            return self.llm_fallback(text), {"intent": "fallback", "confidence": score}

        tag = self.tags[idx]
        responses = self.intents.get(tag)
        if responses:
            return random.choice(responses), {"intent": tag, "confidence": score}

        return self.llm_fallback(text), {"intent": "fallback", "confidence": score}
