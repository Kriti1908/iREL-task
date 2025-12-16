import fitz, json
from rag_inference import answer

doc = fitz.open("endsem.pdf")
questions = []

for page in doc:
    text = page.get_text()
    for line in text.split("\n"):
        if "Q" in line or "Question" in line:
            questions.append(line)

results = {}

for i, q in enumerate(questions):
    results[f"Q{i+1}"] = answer(q)

with open("results/endsem_answers.json", "w") as f:
    json.dump(results, f, indent=2)
