import os
import time
import csv
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --------------------
# CONFIG
# --------------------
QUERY_DIR = "/Users/jordantaylor/PycharmProjects/computer_vision_fire_plan/data/queries"
TARGET_IMG = "/Users/jordantaylor/PycharmProjects/computer_vision_fire_plan/data/targets/target.png"
OUT_CSV = "gemini_symbol_counts.csv"

# Gemini 3 Pro pricing (USD per token, approximate)
INPUT_PRICE_PER_1M = 2.00
OUTPUT_PRICE_PER_1M = 12.00

# --------------------
# SETUP
# --------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def img_part(path, mime="image/png"):
    with open(path, "rb") as f:
        return types.Part(
            inline_data=types.Blob(
                mime_type=mime,
                data=f.read(),
            )
        )

# --------------------
# RUN
# --------------------
rows = []
start_all = time.time()

for fname in sorted(os.listdir(QUERY_DIR)):
    if not fname.lower().endswith(".png"):
        continue

    symbol = os.path.splitext(fname)[0]
    query_path = os.path.join(QUERY_DIR, fname)

    t0 = time.time()

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[
            "You are doing visual symbol matching.",
            f"The first image is a QUERY symbol named '{symbol}'.",
            "The second image is a TARGET maritime fire plan.",
            "Count how many times the query symbol appears in the target image.",
            "Return ONLY JSON: {\"symbol\":\"<name>\",\"count\":<int>}.",
            img_part(query_path),
            img_part(TARGET_IMG),
        ],
    )

    elapsed = time.time() - t0

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count

    cost = (
        (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M +
        (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    )

    rows.append([
        symbol,
        response.text.strip(),
        input_tokens,
        output_tokens,
        round(cost, 6),
        round(elapsed, 3),
    ])

total_time = round(time.time() - start_all, 3)

# --------------------
# WRITE CSV
# --------------------
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "symbol",
        "model_response",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "query_time_sec",
    ])
    writer.writerows(rows)

print(f"Done. Queries: {len(rows)} | Total time: {total_time}s | CSV: {OUT_CSV}")
