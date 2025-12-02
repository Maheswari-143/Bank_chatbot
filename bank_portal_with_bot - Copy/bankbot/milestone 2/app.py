from flask import Flask, render_template, request, jsonify
from chatbot_model import BankBotModel

app = Flask(__name__)
bot = BankBotModel()

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    user_message = request.json['message']
    result = bot.get_response(user_message)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
