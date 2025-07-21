from RetrievalAgent import WikipediaRetriever
from CorrectionAgent import Correction
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from GenratorAgent import Generator
from EmbeddingAgent import Embedding

# parametres partagés
class State(TypedDict, total=False):
    query: str
    corrected_query: str
    article: dict
    status: str
    safe_query: str

# agent de correction 
def correction_agent(state: State) -> State:
    corrected = Correction().run(state["query"])
    if corrected is None:
        print(corrected)
        return {"status": "correction_failed"}
    return {
        "corrected_query": corrected,
        "status": "correction_ok"
    }

# agent retriever
def retriever_agent(state: State) -> State:
    retriever = WikipediaRetriever(lang='en')
    article = retriever.fetch_article(state["corrected_query"])
    if not article.get("raw_text"):
        return {"status": "retrieval_failed"}
    return {"article": article, "status": "retrieval_ok"}

# reoutage et conditionnement entre le noeud end et noeud Retriever
def route_after_correction(state: State) -> str:
    if state["status"] == "correction_ok":
        return "Retriever"
    return "end"

def route_after_retriever(state: State) -> str:
    if state["status"] == "retrieval_ok":
        return "reporter"
    return "end"

def reporter_agent(state: State) -> State:
    report = Generator()
    report.resume(state["article"]["raw_text"], state["corrected_query"])
    return {"status": "report_generated"}

def end_agent(state: State) -> State:
    print(f"[Fin] Status final : {state.get('status')}")
    return state

def embedding_agent(state: State) -> State:
    splitter_agent = Embedding()
    safe_query = splitter_agent.embed_and_index(state["article"]["raw_text"], state["corrected_query"])
    return {"safe_query": safe_query}


def answering_agent(state):
    from AnswerAgent import Answer  # ou l’endroit où est défini Answer
    agent = Answer()
    question = input("\nPosez votre question (ou 'stop' pour terminer) : ")
 # etat finale
    if question.strip().lower() in {"stop", "exit", "quit"}:
        state["status"] = "end"
        return state
    safe_query = state.get("safe_query") 
    print (safe_query)
    answers = agent.answer_question(question, safe_query)
    final_response = agent.generate_llm_response(question, answers)

    print(f"\nQuestion : {question}")
    print("Réponses par RAG :")
    for a in answers:
        print("→", a)
    print("\nRéponse finale générée par LLM :")
    print("→", final_response)
    return state



builder = StateGraph(State)
#noeuds
builder.add_node("Correction", correction_agent)
builder.add_node("Retriever", retriever_agent)
builder.add_node("reporter", reporter_agent)
builder.add_node("embedding", embedding_agent)
builder.add_node("end", end_agent)
builder.add_node("answering", answering_agent)


#etat intiale
builder.set_entry_point("Correction")
#relier
builder.add_conditional_edges("Correction", route_after_correction)
builder.add_conditional_edges("Retriever", route_after_retriever)
builder.add_edge("reporter", "embedding")
builder.add_edge("embedding", "answering")
builder.add_conditional_edges("answering",lambda state: "end" if state.get("status") == "end" else "answering")
graph = builder.compile()
user_query = input("Entrez votre requête Wikipedia : ")
initial_state = {"query": user_query}
final_state = graph.invoke(initial_state)

# Afficher le résultat
print("Requête corrigée :", final_state["corrected_query"])
if "article" in final_state:
    print("Résumé de l'article :", final_state["article"]["raw_text"][:300])
else:
    print("Aucun article récupéré, statut:", final_state.get("status"))


