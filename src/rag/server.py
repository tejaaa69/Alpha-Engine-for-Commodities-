"""
src/rag/server.py

The Flask microservice for the RAG Pipeline.
Run this script in a separate terminal: `python src/rag/server.py`
"""

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from loguru import logger

# Import the clean pipeline we just built
from src.rag.pipeline import RAGPipeline

load_dotenv()

app = Flask("rag_microservice")

# Initialize the pipeline globally
try:
    rag_pipeline = RAGPipeline()
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

@app.route("/health", methods=["GET"])
def health():
    if not rag_pipeline:
        return jsonify({"status": "error", "message": "Pipeline not initialized"}), 500
    return jsonify({
        "status": "ok",
        "nodes_loaded": len(rag_pipeline.nodes),
    })

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = rag_pipeline.query(data["query"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.route("/upload", methods=["POST"])
def handle_upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    upload_dir = "data/rag_data_live"
    os.makedirs(upload_dir, exist_ok=True)

    # Clean old files
    for old in os.listdir(upload_dir):
        os.remove(os.path.join(upload_dir, old))

    saved = []
    for f in request.files.getlist("files"):
        path = os.path.join(upload_dir, f.filename)
        f.save(path)
        saved.append(f.filename)

    try:
        new_count = rag_pipeline.rebuild_from_pdfs(upload_dir)
        return jsonify({"status": "rebuilt", "files": saved, "nodes_built": new_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting RAG Microservice on port 5001...")
    # debug=False and use_reloader=False are important for production stability
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)