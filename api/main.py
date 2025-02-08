from ai_pipeline import query_agent

if __name__ == "__main__":
    print("\n🚀 AI Assistant Ready! Type your query below:\n")

    while True:
        user_query = input("📝 Your Query: ")

        if user_query.lower() in ["exit", "quit"]:
            print("\n👋 Exiting AI Assistant. Have a great day!\n")
            break

        response = query_agent(user_query)
        print("\n💡 Final Answer:\n", response, "\n")
