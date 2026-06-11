import json
import sys


def try_float(x):
    try:
        return float(x)
    except:
        return -1


prices_json_path = sys.argv[1]
asr_prices_json_path = sys.argv[2]
output_csv_path = sys.argv[3]

with open(prices_json_path) as f:
    prices = json.load(f)

with open(asr_prices_json_path) as f:
    asr_prices = json.load(f)["commercial_apis"]

rows = [
    [
        "Modelo",
        "Fornecedor",
        "1M de tokens (entrada)",
        "1M de tokens (saída)",
        "1h de áudio",
    ]
]
for model_name, prices_data in prices.items():
    provider = "SERPRO"
    if "gemini" in model_name:
        provider = "GCP"
    elif "gpt" in model_name and "oss" not in model_name:
        provider = "Microsoft Azure"
    elif "o3-" in model_name or "o4-" in model_name:
        provider = "Microsoft Azure"
    if "dollars_1m_tokens_in" in prices_data and "dollars_1m_tokens_out" in prices_data:
        row = [
            model_name,
            provider,
            float(prices_data["dollars_1m_tokens_in"]),
            float(prices_data["dollars_1m_tokens_out"]),
            None,
        ]
        rows.append(row)
for model_name, price in asr_prices.items():
    row = [
        model_name,
        "Microsoft Azure" if "azure" in model_name else "GCP",
        None,
        None,
        float(price),
    ]
    rows.append(row)

rows.sort(
    key=lambda x: (
        x[1],
        (0 if x[2] is None else x[2]) + (0 if x[3] is None else x[3]),
        0 if x[4] is None else x[4],
    )
)

with open(output_csv_path, "w") as f:
    for row in rows:
        num1 = str(row[2]).replace(".", ",")
        num2 = str(row[3]).replace(".", ",")
        num3 = str(row[4]).replace(".", ",")
        f.write("\t".join([row[0], row[1], num1, num2, num3]) + "\n")
