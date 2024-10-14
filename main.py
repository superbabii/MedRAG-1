import json
import random
from src.medrag import MedRAG

# Load the benchmark JSON file
with open('benchmark.json', 'r') as f:
    benchmark_data = json.load(f)

# Get 5 random questions
random_questions = random.sample(list(benchmark_data.items()), 3)

medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="PubMed")

# Store the results of comparisons
results = []
correct_count = 0

# Iterate over each question and get the generated answer
for question_id, question_data in random_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    # Use MedRAG to generate the answer
    answer, snippets, scores = medrag.answer(question=question, options=options, k=3)

    # Parse the generated answer and compare with correct answer
    generated_answer_dict = json.loads(answer)
    generated_choice = generated_answer_dict.get('answer_choice', None)

    # Compare the generated answer with the correct one
    is_correct = correct_answer == generated_choice[0]
    if is_correct:
        correct_count += 1

    result = {
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_choice,
        'is_correct': is_correct,
        'snippets': snippets,
        'scores': scores
    }
    results.append(result)

# Print the results of the comparison
for result in results:
    print(f"Score: {result['scores']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print('-' * 50)

# Calculate accuracy
accuracy = correct_count / len(results) * 100
print(f"Accuracy: {accuracy:.2f}%")
