# MultiAgent_RAG_SYSTEM

## Description

Ce projet implémente un système RAG (Retrieval-Augmented Generation) multi-agents basé sur `langgraph` pour l'extraction, la vectorisation et l'interrogation de contenus Wikipedia. Il permet de :
- Sélectionner un sujet (topic) donné par l'utilisateur.
- Corriger et reformuler ce sujet via un LLM.
- Extraire l'article Wikipedia correspondant via une API.
- Générer un rapport PDF résumant l'article.
- Vectoriser et indexer le contenu pour permettre un questionnement ultérieur (RAG).
- Répondre à toute question sur le sujet en utilisant la recherche sémantique et un LLM.

## Fonctionnalités principales

- **CorrectionAgent** : Corrige et reformule la requête utilisateur pour optimiser la recherche Wikipedia.
- **RetrievalAgent** : Récupère l'article Wikipedia correspondant au sujet corrigé.
- **GenratorAgent** : Génère un rapport PDF résumant l'article Wikipedia.
- **EmbeddingAgent** : Découpe l'article en sections, vectorise les phrases et les indexe dans FAISS pour la recherche sémantique.
- **AnswerAgent** : Permet de poser des questions sur le sujet, recherche les passages les plus pertinents et génère une réponse concise via LLM.

## Installation

1. **Cloner le dépôt**  
   ```bash
   git clone <url_du_repo>
   cd projet_wikipedia
   ```

2. **Installer les dépendances**  
   (Assurez-vous d'avoir Python 3.8+ et pip installés)
   ```bash
   pip install -r requirements.txt
   ```
   Les principales dépendances sont :  
   - `langgraph`
   - `wikipedia-api`
   - `faiss`
   - `sentence-transformers`
   - `transformers`
   - `fpdf`
   - `langchain_ollama`
   - `torch`
   - `scikit-learn`
   - etc.

3. **Télécharger le modèle de police**  
   Le fichier `DejaVuSans.ttf` doit être présent dans le dossier pour la génération de PDF.

## Utilisation

### 1. Exécution du graphe complet (recommandé)
Lancez le script principal pour démarrer le flux multi-agents :
```bash
python Langgrph.py
```
Vous serez invité à entrer un sujet Wikipedia, puis à poser des questions sur ce sujet.

### 2. Exécution étape par étape (pour tests)
Vous pouvez aussi exécuter `main.py` pour tester chaque étape individuellement.

## Exemple de flux

1. **Entrée utilisateur** : "lionel messi"
2. **Correction** : Reformulation du sujet si besoin.
3. **Récupération** : Extraction de l'article Wikipedia.
4. **Résumé PDF** : Génération d'un rapport PDF.
5. **Vectorisation** : Indexation des phrases pour la recherche.
6. **Question/Réponse** : Posez une question, le système retrouve les passages pertinents et génère une réponse.

## Fichiers générés

- `rapport_<topic>.pdf` : Rapport PDF résumé.
- `<topic>_sections.index` et `<topic>_texts.pkl` : Index FAISS et textes pour la recherche sémantique.

## Remarques

- Le projet nécessite une carte GPU pour accélérer l'inférence (optionnel mais recommandé).
- Le modèle LLM utilisé par défaut est `mistral:7b` via `langchain_ollama`.
- Les réponses sont strictement basées sur le contenu Wikipedia extrait.

## Auteurs

- Hazem Jday 