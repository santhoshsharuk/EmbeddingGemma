from flask import Flask, request, jsonify
from pdf_utils import extract_text_from_pdf, split_text
from vector_store import add_chunks, search_query
import os

app = Flask(__name__)
os.makedirs("embeddings", exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files["pdf"]
    text = extract_text_from_pdf(file)
    chunks = split_text(text)
    add_chunks(chunks)
    return f"PDF uploaded and processed! Added {len(chunks)} chunks."

@app.route("/")
def home():
    return "Welcome to the PDF processing API!"

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("question")
    results = search_query(query, top_k=3)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
