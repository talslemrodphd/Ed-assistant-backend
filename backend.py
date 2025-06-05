import os
from flask import Flask, request, jsonify
import dropbox
import openai

app = Flask(__name__)

DROPBOX_TOKEN = os.getenv("DROPBOX_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

dbx = dropbox.Dropbox(DROPBOX_TOKEN)
openai.api_key = OPENAI_API_KEY

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")

    # For now, just echo the question as a placeholder
    answer = f"You asked: {question}"

    return jsonify({"answer": answer, "sources": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
