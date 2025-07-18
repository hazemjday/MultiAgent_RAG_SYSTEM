import subprocess
import re

def query_mistral(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral:7b", prompt],
        capture_output=True,
        text=True,
        encoding='utf-8'   # <<< Important ici !
    )
    if result.stderr:
        print("Erreur :", result.stderr)
    return result.stdout.strip()

def extract_corrected_title(text):
    match = re.search(r'“([^”]+)”|\"([^\"]+)\"', text)
    if match:
        return match.group(1) or match.group(2)
    return None
prompt = 'Correct and return the Wikipedia search query only between double quotes, without explanation or definition, in a single line: squid game'
response = query_mistral(prompt)
print("Réponse complète :", response)

corrected_title = extract_corrected_title(response)
print("Titre corrigé extrait :", corrected_title)
