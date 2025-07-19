from sentence_transformers import SentenceTransformer

# Chargez le modèle
model = SentenceTransformer('all-MiniLM-L6-v2')  # Exemple

# Obtenez la taille des embeddings
print(f"Taille des embeddings : {model.get_sentence_embedding_dimension()} dimensions")
