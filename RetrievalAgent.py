import wikipediaapi

class WikipediaRetriever:
    def __init__(self, lang='fr', user_agent='MonProjetRAG/1.0 (contact@example.com)'):
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent
        )

    def fetch_article(self, topic):
        page = self.wiki.page(topic)
        
        if not page.exists() :
            return {
                "topic": topic,
                "raw_text": None,
                "status": "not_found_or_too_short"
            }
        
        return {
            "topic": topic,
            "raw_text": page.text,
            "status": "ok"
        }