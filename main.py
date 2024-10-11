from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Replace with your actual API key and endpoint
GEMINI_API_URL = "https://gemini-api-url.com/ask"  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('query')
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": user_input,
        "max_tokens": 150,
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        response.raise_for_status()
        answer = response.json()['choices'][0]['text']
        return jsonify({"answer": answer})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
