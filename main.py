from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}

medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="PubMed")
answer, snippets, scores = medrag.answer(question=question, options=options, k=5)

# Printing the results
print("Answer:", answer)
print("\nSnippets:")
for i, snippet in enumerate(snippets):
    print(f"Snippet {i+1}: {snippet}")
    
print("\nScores:")
for i, score in enumerate(scores):
    print(f"Score {i+1}: {score}")