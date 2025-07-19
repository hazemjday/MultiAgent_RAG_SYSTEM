import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class Answer:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[AnswerAgent] Using device: {self.device}")
        self.model = SentenceTransformer(model_name).to(self.device)

    def answer_question(self, question: str, safe_query: str, top_k: int = 3) -> list:
        # Charger index et textes
        index = faiss.read_index(f"{safe_query}_sections.index")
        with open(f"{safe_query}_texts.pkl", "rb") as f:
            texts = pickle.load(f)

        # Encoder la question
        question_embedding = self.model.encode(question)
        normalized_embedding = normalize([question_embedding], norm='l2')[0]

        # Rechercher dans l'index
        D, I = index.search(np.array([normalized_embedding], dtype='float32'), top_k)

        # Retourner les réponses correspondantes
        answers = [texts[i] for i in I[0]]
        return answers