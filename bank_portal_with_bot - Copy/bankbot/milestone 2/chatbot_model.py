# chatbot_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BankBotModel:
    def __init__(self, dataset_path="chatbot_dataset.csv"):
        data = pd.read_csv(dataset_path)
        self.data = data[["text", "intent", "response"]]
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.data["text"])

    def get_response(self, user_input):
        user_vec = self.vectorizer.transform([user_input.lower()])
        similarity = cosine_similarity(user_vec, self.X)
        idx = similarity.argmax()

        if similarity.max() < 0.2:
            return {"intent": "out_of_scope", "response": "🤔 Sorry, I don’t have an answer for that question."}
        else:
            intent = self.data.iloc[idx]["intent"]
            response = self.data.iloc[idx]["response"]
            return {"intent": intent, "response": response}
