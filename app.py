"""
Flask web server for RAG Study Assistant.
"""

import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag = RAGEngine(use_local_llm=os.getenv("USE_LOCAL_LLM", "0") == "1")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. PDF only."}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    try:
        n_chunks = rag.ingest_pdf(path)
        return jsonify({"message": f"Ingested {filename} -> {n_chunks} chunks indexed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    try:
        result = rag.ask(question)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/reset", methods=["POST"])
def reset():
    rag.reset_memory()
    return jsonify({"message": "Conversation memory cleared."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
