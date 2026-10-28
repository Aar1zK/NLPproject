"""
Flask Web Server & REST API for the Classical NLP Study Assistant.
Supports:
  - Preset textbook chapters
  - Custom pasted text / notes
  - Custom file uploads (.txt, .md, .pdf)
  - Real-time execution & mathematical inspection of all 4 classical NLP algorithms
"""

import io
import json
import os
from flask import Flask, render_template, request, jsonify
from nlp_engine.pipeline import StudyAssistantPipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CHAPTERS_PATH = os.path.join(BASE_DIR, "data", "sample_chapters.json")

pipeline = StudyAssistantPipeline()


def load_sample_chapters():
    if os.path.exists(SAMPLE_CHAPTERS_PATH):
        with open(SAMPLE_CHAPTERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.route("/")
def index():
    chapters = load_sample_chapters()
    return render_template("index.html", chapters=chapters)


@app.route("/api/chapters", methods=["GET"])
def get_chapters():
    return jsonify({"chapters": load_sample_chapters()})


@app.route("/api/analyze", methods=["POST"])
def analyze_document():
    """
    Runs Algorithm 1 (TextRank Summarization), Algorithm 2 (HMM + Viterbi IE),
    and builds the index for Algorithm 3 (Okapi BM25) on any preset or custom text.
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    title = (data.get("title") or "Custom Study Document").strip()
    summary_k = int(data.get("summary_k", 4))
    damping_factor = float(data.get("damping_factor", 0.85))

    if not text:
        return jsonify({"error": "Please provide or upload non-empty document text."}), 400

    result = pipeline.analyze_document(
        text=text,
        title=title,
        summary_k=summary_k,
        damping_factor=damping_factor
    )
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Extracts text from a user-uploaded .txt, .md, or .pdf file and immediately
    runs the unified 4-algorithm pipeline on it.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    extracted_text = ""

    try:
        if ext in (".txt", ".md", ".csv", ".log"):
            raw_bytes = file.read()
            extracted_text = raw_bytes.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file.read()))
                pages_text = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                extracted_text = "\n\n".join(pages_text)
            except Exception as pdf_err:
                return jsonify({"error": f"Failed to read PDF file: {str(pdf_err)}"}), 400
        else:
            return jsonify({"error": f"Unsupported file extension '{ext}'. Please upload .txt, .md, or .pdf."}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing uploaded file: {str(e)}"}), 500

    extracted_text = extracted_text.strip()
    if not extracted_text:
        return jsonify({"error": "The uploaded file contained no extractable text."}), 400

    summary_k = int(request.form.get("summary_k", 4))
    damping_factor = float(request.form.get("damping_factor", 0.85))

    result = pipeline.analyze_document(
        text=extracted_text,
        title=f"Uploaded: {filename}",
        summary_k=summary_k,
        damping_factor=damping_factor
    )
    result["raw_text"] = extracted_text
    return jsonify(result)


@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Runs Algorithm 3 (Okapi BM25 Passage Retrieval) and Algorithm 4
    (Multinomial Naive Bayes EAT Classification + HMM Span Pinpointing + Lesk WSD)
    to answer any natural language question about the active document.
    """
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    text = data.get("text")
    k1 = float(data.get("k1", 1.5))
    b = float(data.get("b", 0.75))

    if not question:
        return jsonify({"error": "Please enter a question to ask."}), 400

    result = pipeline.ask_question(
        question=question,
        text=text,
        k1=k1,
        b=b
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
