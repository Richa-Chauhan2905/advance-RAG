from parallel_query import parallel_query, get_final_answer

def main():
    while True:
        query = input("> ")
        if query.lower() == "clear":
            break

        retrieved_chunks = parallel_query(query)

        if not retrieved_chunks:
            print("No relevant results found")
        else:
            final_ans = get_final_answer(query, retrieved_chunks)

            print(f"\nRelevant Result:\n{final_ans}")

if __name__ == "__main__":
    main()
