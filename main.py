from wiki import WikipediaRetriever
from correction import CorrectionAgent


user_query = "iphon 16"
correction_agent = CorrectionAgent()
corrected_query = correction_agent.run(user_query)
if corrected_query is None:
        print("La requête corrigée est introuvable.")
else:
        print("Requête corrigée :", corrected_query)

retriever = WikipediaRetriever(lang='en')
result = retriever.fetch_article(corrected_query)

if result["status"] == "ok":
    print(result["raw_text"])
else:
    print("Article indisponible ou trop court.")