from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="In two sentences, what is an indemnification clause?",
    generation_config={"thinking_level": "low", "max_output_tokens": 300},
)

print(interaction.output_text)
print(interaction.usage)