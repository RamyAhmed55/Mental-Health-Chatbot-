from flask import Blueprint, render_template, request, jsonify
from .pipeline_manager import pipeline

main = Blueprint("main", __name__)

# =====================================================================================================

@main.route("/")
def index():
    return render_template("index.html")

# =====================================================================================================

@main.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "empty message"}), 400

    result = pipeline.chat(message)
    return jsonify(result)