from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from werkzeug.utils import secure_filename

# -----------------------
# Configuration
# -----------------------

UPLOAD_FOLDER = "uploaded_pdfs"
PROJECTS_FOLDER = "projects_folder"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROJECTS_FOLDER, exist_ok=True)

# ⚠️ IMPORTANT: Put your API key here safely
import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


app = Flask(__name__)
CORS(app)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadata = []

# -----------------------
# Load Initial PDFs
# -----------------------

for file in os.listdir(PROJECTS_FOLDER):
    if file.endswith(".pdf"):
        with pdfplumber.open(os.path.join(PROJECTS_FOLDER, file)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            documents.append(text)
            metadata.append(file)

print("Loaded documents:", len(documents))

# -----------------------
# Create FAISS Index
# -----------------------

if len(documents) > 0:
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
else:
    dimension = 384  # MiniLM default
    index = faiss.IndexFlatL2(dimension)

# -----------------------
# Routes
# -----------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/files", methods=["GET"])
def get_files():
    return jsonify({
        "files": metadata
    })

from flask import send_from_directory

@app.route("/preview/<path:filename>")
def preview_file(filename):
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    project_path = os.path.join(PROJECTS_FOLDER, filename)

    # Check uploaded_pdfs first
    if os.path.exists(upload_path):
        return send_from_directory(UPLOAD_FOLDER, filename)

    # Then check projects_folder
    if os.path.exists(project_path):
        return send_from_directory(PROJECTS_FOLDER, filename)

    return "File not found", 404

@app.route("/search", methods=["POST"])
def search():
    try:
        if len(documents) == 0:
            return jsonify({
                "student_name": "No PDFs available",
                "description": "Upload a project PDF first."
            })

        data = request.json
        query = data.get("query", "").strip()

        if not query:
            return jsonify({
                "student_name": "No Match Found",
                "description": ""
            })

        # 🔎 Encode query
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), k=1)

        best_distance = float(distances[0][0])
        best_index = indices[0][0]

        print("Distance:", best_distance)  # optional debug

        # 🔥 THRESHOLD CHECK
        # Higher distance = less similar
        if best_distance > 1.6:
            return jsonify({
                "student_name": "No Match Found",
                "description": ""
            })

        best_match = documents[best_index]
        file_name = metadata[best_index]

        limited_text = best_match[:2000]

        # 🤖 Ask Groq to summarize
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
From the following hackathon project text extract:

1. Project Name
2. Team Members (Names only)
3. Clear 4-5 line summary explaining:
   - What problem it solves
   - How it works
   - Main benefit

Keep it clean and structured.
No long paragraphs.
No extra explanation.

Return format:

Project Name:
Team Members:
Summary:

Text:
{limited_text}
"""
                    }
                ]
            )

            summary = response.choices[0].message.content

        except Exception as groq_error:
            print("Groq Error:", repr(groq_error))
            summary = "Summary generation failed."

        return jsonify({
            "student_name": file_name,
            "description": summary
        })

    except Exception as e:
        print("ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    try:
        global documents, metadata, index

        if filename not in metadata:
            return jsonify({"error": "File not found"}), 404

        # Get index of file
        file_index = metadata.index(filename)

        # Remove from lists
        metadata.pop(file_index)
        documents.pop(file_index)

        # Delete physical file
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        # Rebuild FAISS index
        if len(documents) > 0:
            embeddings = model.encode(documents)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))
        else:
            index = faiss.IndexFlatL2(384)

        return jsonify({"message": f"{filename} deleted successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)

        # 🔥 DUPLICATE NAME CHECK
        if filename in metadata:
            return jsonify({"error": "Duplicate file name already exists!"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        if not text.strip():
            return jsonify({"error": "PDF has no readable text"}), 400

        new_embedding = model.encode([text])

        # 🔥 DUPLICATE CONTENT CHECK
        if len(documents) > 0:
            D, I = index.search(np.array(new_embedding), k=1)
            similarity_score = float(D[0][0])

            if similarity_score < 0.5:  # small distance = very similar
                os.remove(filepath)
                return jsonify({"error": "Duplicate project content detected!"}), 400

        documents.append(text)
        metadata.append(filename)
        index.add(np.array(new_embedding))

        return jsonify({"message": f"{filename} uploaded successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
