import re
from langchain_ollama import OllamaLLM

class CorrectionAgent:
    def __init__(self, model="mistral:7b", temperature=0):
        # Initialisation du modèle Ollama
        self.llm = OllamaLLM(model=model, temperature=temperature)

    def extract_between_quotes(self, text):
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1)
        return None

    def correct_query_with_llm(self, query):
        prompt = (
            f'Correct and return the Wikipedia search query only between double quotes, '
            f'without explanation or parentheses, in a single line: {query}'
        )
        response = self.llm.invoke(prompt)
        return response.strip()

    def run(self, query):
        corrected = self.correct_query_with_llm(query)
        extracted = self.extract_between_quotes(corrected)
        return extracted