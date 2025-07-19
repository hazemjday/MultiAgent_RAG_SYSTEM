from RetrievalAgent import WikipediaRetriever
from CorrectionAgent import Correction
from EmbeddingAgent import Embedding
from AnswerAgent import Answer
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import torch



user_query = "elon musk"
correction_agent = Correction()
corrected_query = correction_agent.run(user_query)
if corrected_query is None:
        print("La requête corrigée est introuvable.")
else:
        print("Requête corrigée :", corrected_query)

retriever = WikipediaRetriever(lang='en')
result = retriever.fetch_article(corrected_query)
if result["status"] == "ok":
    print("success")
else:
    print("Article indisponible ou trop court.")

#division du paragraphe wikipidia en sections
splitter_agent = Embedding()
safe_query = splitter_agent.embed_and_index(result["raw_text"], corrected_query)
print(safe_query)

agent = Answer()

question = "does musk bought twitter"
answers = agent.answer_question(question, safe_query)
print(f"\nQuestion : {question}\nRéponses probables :")
for ans in answers:
    print("→", ans)