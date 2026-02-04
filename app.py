# ---------------- Imports ----------------
import os
import logging
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ---------------- Load Environment ----------------
load_dotenv()
# Ensure VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "conversations.db")
INTENTS_FILE = os.path.join(BASE_DIR, "nlp", "intents.json")
CSV_FILE = os.path.join(BASE_DIR, "nlp", "data.csv")  # optional

# ---------------- Logging ----------------
LOG_PATH = os.path.join(BASE_DIR, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ---------------- Flask App ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ---------------- NLP Pipeline ----------------
try:
    from nlp.pipeline import NLPPipeline
    try:
        nlp = NLPPipeline(
            csv_file=CSV_FILE,
            intents_file=INTENTS_FILE,
            openai_model=None,
            ollama_model="gemma:2b"
        )
        logging.info("NLPPipeline initialized successfully.")
    except Exception as e:
        logging.exception("Error initializing NLPPipeline; using stub. Error: %s", e)

        class _StubNLPPipeline:
            def get_response(self, text: str):
                return (
                    "Sorry — I'm temporarily unable to access the knowledge service. "
                    "Please try again later.",
                    {"intent": "fallback", "confidence": 0.0}
                )
        nlp = _StubNLPPipeline()
except Exception as e:
    logging.exception("Failed to import NLPPipeline; using stub. Error: %s", e)

    class _StubNLPPipeline:
        def get_response(self, text: str):
            return (
                "Sorry — I'm temporarily unable to access the knowledge service. "
                "Please try again later.",
                {"intent": "fallback", "confidence": 0.0}
            )
    nlp = _StubNLPPipeline()

# ---------------- Sentiment Analyzer ----------------
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logging.exception("Failed to initialize SentimentIntensityAnalyzer: %s", e)

    class _StubSia:
        def polarity_scores(self, _):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    sia = _StubSia()

# ---------------- Database ----------------
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                direction TEXT,
                text TEXT,
                intent TEXT,
                confidence REAL,
                sentiment TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database initialized at %s", DB_PATH)
    except Exception as e:
        logging.exception("Failed to initialize DB: %s", e)

def log_message(direction: str, text: str, meta: dict = None):
    try:
        meta = meta or {}
        intent = meta.get("intent")
        conf = meta.get("confidence")
        sentiment = meta.get("sentiment")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        ts = datetime.utcnow().isoformat() + "Z"
        c.execute(
            "INSERT INTO messages (timestamp, direction, text, intent, confidence, sentiment) VALUES (?,?,?,?,?,?)",
            (ts, direction, text, intent, conf, str(sentiment))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.exception("Failed to log message to DB: %s", e)

init_db()

# ---------------- Routes ----------------
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/chat")
def chat():
    return render_template("index.html")

@app.route("/journal")
def journal():
    return render_template("journal.html")

@app.route("/community")
def community():
    return render_template("community.html")

@app.route("/quiz")
def quiz():
    return render_template("quiz.html")

@app.route("/meditations")
def meditations():
    return render_template("meditations.html")

@app.route("/therapists")
def therapists():
    return render_template("therapists.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/breathing")
def breathing():
    return render_template("breathing.html")

@app.route("/bodyscan")
def bodyscan():
    return render_template("bodyscan.html")

@app.route("/mantra")
def mantra():
    return render_template("manta.html")

@app.route("/walking")
def walking():
    return render_template("walking.html")

@app.route("/visual")
def visual():
    return render_template("visual.html")

@app.route("/kindness")
def kindness():
    return render_template("kind.html")

# ---------------- Chatbot API ----------------
@app.route("/api/message", methods=["POST"])
def message():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # NLP response
    try:
        response_text, meta = nlp.get_response(text)
    except Exception as e:
        logging.exception("nlp.get_response raised exception: %s", e)
        response_text = "Sorry, I'm having trouble generating a response right now."
        meta = {"intent": "error", "confidence": 0.0}

    if not isinstance(meta, dict):
        meta = {"intent": str(meta) if meta else "unknown", "confidence": 0.0}

    # Sentiment
    try:
        sentiment = sia.polarity_scores(text)
    except Exception as e:
        logging.exception("Sentiment analysis failed: %s", e)
        sentiment = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    meta["sentiment"] = sentiment

    # Log user message
    log_message("user", text, meta)

    # Crisis handling
    if sentiment.get("compound", 0.0) < -0.7:
        response_text = (
            "I’m really sorry you’re feeling this way. "
            "If you're in immediate danger, call local emergency services now. "
            "Would you like me to provide hotline numbers or resources?"
        )

    # Log bot response
    log_message("bot", response_text, meta)

    # Recommendation
    compound = sentiment.get("compound", 0.0)
    if compound >= 0.5:
        recommendation = "You seem happy! Keep doing what you enjoy 😊"
    elif 0 <= compound < 0.5:
        recommendation = "You seem okay. Maybe take a short break or relax."
    elif -0.5 <= compound < 0:
        recommendation = "You might be stressed. Try deep breathing or meditation."
    else:
        recommendation = "You seem low. Consider talking to someone you trust."

    return jsonify({
        "response": response_text,
        "meta": meta,
        "recommendation": recommendation
    })

# ---------------- Journal API ----------------
@app.route("/journal/submit", methods=["POST"])
def journal_submit():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No journal text provided"}), 400

    try:
        _, meta = nlp.get_response(text)
    except Exception as e:
        logging.exception("nlp.get_response raised exception (journal): %s", e)
        meta = {"intent": "error", "confidence": 0.0}

    if not isinstance(meta, dict):
        meta = {"intent": "unknown", "confidence": 0.0}

    try:
        sentiment = sia.polarity_scores(text)
    except Exception as e:
        logging.exception("Sentiment analysis failed (journal): %s", e)
        sentiment = {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    meta["sentiment"] = sentiment
    log_message("user", text, meta)

    compound = sentiment.get("compound", 0.0)
    if compound >= 0.5:
        recommendation = "You seem happy! Keep doing what you enjoy 😊"
    elif 0 <= compound < 0.5:
        recommendation = "You seem okay. Maybe take a short break or relax."
    elif -0.5 <= compound < 0:
        recommendation = "You might be stressed. Try deep breathing or meditation."
    else:
        recommendation = "You seem low. Consider talking to someone you trust."

    return jsonify({"meta": meta, "recommendation": recommendation})

# ---------------- Quiz API ----------------
@app.route("/predict", methods=["POST"])
def quiz_submit():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No symptoms submitted"}), 400

    try:
        symptoms = {k: int(v) for k, v in data.items()}
        score = sum(symptoms.values())
    except Exception as e:
        logging.exception("Invalid quiz payload: %s", e)
        return jsonify({"error": "Invalid payload; expected numeric scores."}), 400

    if score <= 5:
        condition, code = "No significant symptoms", 0
    elif 6 <= score <= 10:
        condition, code = "Mild Depression / Anxiety", 1
    elif 11 <= score <= 15:
        condition, code = "Moderate Depression / Generalized Anxiety Disorder", 2
    elif 16 <= score <= 20:
        condition, code = "Severe Depression", 3
    elif 21 <= score <= 25:
        condition, code = "Panic Disorder / Social Anxiety Disorder", 4
    elif 26 <= score <= 30:
        condition, code = "Obsessive-Compulsive Disorder (OCD)", 5
    elif 31 <= score <= 35:
        condition, code = "Bipolar Disorder", 6
    else:
        condition, code = "Post-Traumatic Stress Disorder (PTSD) / Severe Concern", 7

    return jsonify({
        "prediction": code,
        "label": condition,
        "score": score
    })

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
