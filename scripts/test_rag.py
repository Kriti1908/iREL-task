# test_rag.py

from rag_inference import generate_answer

def main():
    print("=" * 60)
    print(" IIIT Course Assistant (Qwen3-4B + LoRA + RAG)")
    print(" Type 'exit' or 'quit' to stop")
    print("=" * 60)

    while True:
        try:
            question = input("\n🧑‍🎓 Ask a question: ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if len(question) == 0:
            print("⚠️  Please enter a valid question.")
            continue

        print("\n🔍 Thinking...\n")

        try:
            result = generate_answer(question, thinking=True)
        except Exception as e:
            print("❌ Error during inference:", e)
            continue

        print("📘 Answer:")
        print("-" * 60)
        print(result["answer"].strip())
        print("-" * 60)

        if result["youtube"]:
            print("\n📺 Suggested YouTube Videos:")
            for link in result["youtube"]:
                print("  •", link)

        if result["papers"]:
            print("\n📄 Relevant Research Papers:")
            for paper in result["papers"]:
                print("  •", paper)

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()