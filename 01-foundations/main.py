
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if not api_key or api_key == "your_actual_api_key":
        raise SystemExit("Please configure GEMINI_API_KEY in .env")

    client = genai.Client(api_key=api_key)
    model_info = client.models.get(model=model)

    print("Maximum input tokens:", model_info.input_token_limit)
    print("Maximum output tokens:", model_info.output_token_limit)

    system_instruction = """
    You are a programming tutor.
    Explain technical concepts clearly with simple examples.
    """

    prompt = "Explain the difference between an API and an SDK."

    try:

        print("Counting prompt tokens...", flush=True)

        token_count = client.models.count_tokens(
            model=model,
            contents=prompt,
        )

        print(
            f"Prompt tokens counted: {token_count.total_tokens}",
            flush=True,
        )

        print("Sending prompt to Gemini...", flush=True)

        start = perf_counter()

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )

        latency = perf_counter() - start

        print("Gemini response received.", flush=True)

    except Exception as exc:
        raise SystemExit(f"Gemini request failed: {exc}") from exc


    print("\n--- MODEL RESPONSE ---")
    print(response.text or "No text response returned.")

    print("\n--- REQUEST METRICS ---")
    print(f"Model: {model}")
    print(f"Prompt-only preflight tokens: {token_count.total_tokens}")
    print(f"Generation latency: {latency:.2f} seconds")

    usage = response.usage_metadata

    if usage is None:
        print("Token usage metadata was unavailable.")
        return

    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    thinking_tokens = usage.thoughts_token_count or 0
    total_tokens = usage.total_token_count or 0

    print(f"Actual input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Thinking tokens: {thinking_tokens}")
    print(f"Total reported tokens: {total_tokens}")

    # Estimate Standard paid-tier token cost in USD.
    # Gemini 3.8 Flash's current introductory rates apply
    # through December 31, 2026.
    input_rate = float(
        os.getenv("GEMINI_INPUT_PRICE_PER_MILLION_USD", "0.75")
    )
    output_rate = float(
        os.getenv("GEMINI_OUTPUT_PRICE_PER_MILLION_USD", "3.75")
    )

    estimated_cost = (
        input_tokens * input_rate
        + (output_tokens + thinking_tokens) * output_rate
    ) / 1_000_000

    print(f"Estimated paid-tier cost: ${estimated_cost:.8f} USD")


if __name__ == "__main__":
    main()
