from google import genai

client = genai.Client()
prompt = "Name one risk in a contract with no liability cap. One sentence."

for run in range(5):
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        generation_config={"thinking_level": "low", "max_output_tokens": 100},
    )
    usage = interaction.usage
    print(f"[{usage.total_input_tokens} in / {usage.total_output_tokens} out] {interaction.output_text}")