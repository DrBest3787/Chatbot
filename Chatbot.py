from transformers import pipeline

def text_bot():
    print("Welcomme. I´m your CoPilot. Ask me something.")
    print("Say 'exit' to quit")

    chatbot = pipeline("text-generation", model="gpt2", max_length=100)

    while True:
        # Benutzereingabe
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Bot: Bye!")
            break
        try:
            response = chatbot(user_input)

            print("Bot:", response[0]['generated_text'])
        except Exception as e:
            print(f"An error has been detected: {e}")

# Main-Funktion starten
if __name__ == "__main__":
    text_bot()
