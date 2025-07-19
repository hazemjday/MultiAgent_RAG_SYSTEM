import torch
import faiss
import pickle
import re
import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class Embedding:
    def __init__(self, min_length=50):
        self.min_length = min_length
        self.exclude_titles = [
            "External links", "References", "Further reading", 
            "See also", "Footnotes", "Bibliography"
        ]
        self.section_pattern = re.compile(r'^[A-Z][A-Za-z0-9\s]+[^\.,;:?!]$')

    def split_into_sections(self, text: str) -> Dict[str, str]:
        lines = text.split('\n')
        sections = {}
        current_section = "Introduction"
        sections[current_section] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self.section_pattern.match(line):
                current_section = line
                sections[current_section] = []
            else:
                sections[current_section].append(line)

        for sec in list(sections.keys()):
            sections[sec] = " ".join(sections[sec])
            if not sections[sec].strip() or sec in self.exclude_titles or len(sections[sec]) < self.min_length:
                del sections[sec]
        return sections


    def embed_and_index(self, raw_text: str, corrected_query: str) -> str:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Utilisation du device : {device}")
        model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        texts = []
        sections = self.split_into_sections(raw_text)
        for section_name, paragraphs in sections.items():
            phrases = [phrase.strip() for phrase in paragraphs.split('.') if phrase.strip()]
            for phrase in phrases:
                phrase_embedding = model.encode(phrase)
                normalized_embedding = normalize([phrase_embedding], norm='l2')[0]
                index.add(np.array([normalized_embedding], dtype='float32'))
                texts.append(phrase)

        safe_query = re.sub(r'\W+', '_', corrected_query.lower())
        faiss.write_index(index, f"{safe_query}_sections.index")
        with open(f"{safe_query}_texts.pkl", "wb") as f:
            pickle.dump(texts, f)

        print(f"Nombre de phrases indexées : {index.ntotal}")
        return safe_query