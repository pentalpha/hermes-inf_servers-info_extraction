import os
import sys
import json


if __name__ == "__main__":
    results_files = sys.argv[1:]

    # Load prices
    try:
        with open("input/test_prices.json", "r") as f:
            prices = json.load(f)
            standard_price_per_hour = prices["gcp_n1_t4"]["dollars_hour_standard"]
    except FileNotFoundError:
        print("Error: input/test_prices.json not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in prices JSON: {e}")
        sys.exit(1)

    data = []

    specific_fmax_categories = [
        "Fato Ocorrendo Neste Momento ?",
        "Autor Do Fato No Local ?",
        "Autor Do Fato Armado ?",
        "Feridos Com Risco de Morte ?",
        "Risco De Tumulto ?",
        "Lei Maria da Penha ?",
    ]

    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"Warning: File {results_file} not found. Skipping.")
            continue

        with open(results_file, "r") as f:
            try:
                file_content = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {results_file}. Skipping.")
                continue

        # Handle if the file contains a list of results or a single dict
        if isinstance(file_content, dict):
            items = [file_content]
        elif isinstance(file_content, list):
            items = file_content
        else:
            print(f"Warning: Unexpected content format in {results_file}. Skipping.")
            continue

        for item in items:
            row = {}

            # Model names
            full_model_name = item.get("model", "Unknown")
            row["full_model"] = full_model_name

            if " + " in full_model_name:
                parts = full_model_name.split(" + ")
                row["ner_model"] = parts[0]
                row["classification_model"] = parts[1]
            else:
                row["ner_model"] = full_model_name
                row["classification_model"] = full_model_name

            # Mean Metrics
            # mean_fmax (specific categories)
            fmax_dict = item.get("fmax", {})
            fmax_values = [
                fmax_dict.get(cat)
                for cat in specific_fmax_categories
                if cat in fmax_dict
            ]

            if fmax_values and len(fmax_values) > 0:
                row["mean_fmax"] = sum(fmax_values) / len(fmax_values)
            else:
                row["mean_fmax"] = None  # Or 0 depending on preference, None is safer

            # mean_jw_sim
            row["mean_jw_sim"] = item.get("mean_metrics", {}).get("jw_sim")

            row["mean_quality"] = (row["mean_fmax"] + row["mean_jw_sim"]) / 2

            # Categories (fmax for questions, jw_sim for labels)
            # Assumption: Questions end with "?" -> fmax. Others -> jw_sim.
            # The prompt lists specific categories for mean_fmax, but also says:
            # "For the categories ("* ?"), it should be fmax. For the labels, it should be jw_sim"
            # We will iterate over keys available in the item's fmax/jw_sim dictionaries to populate columns.

            # Populate from fmax dict (for questions)
            for k, v in item.get("fmax", {}).items():
                if "?" in k:
                    row[k] = v

            # Populate from jw_sim dict (for labels - not questions)
            for k, v in item.get("jw_sim", {}).items():
                if "?" not in k:
                    row[k] = v

            # Metadata & Costs
            meta = item.get("meta", {})
            samples = meta.get("samples", 446)
            gpu_seconds = meta.get("gpu_seconds", 0)
            tokens_in_sec = meta.get("tokens_per_second_in", 0)
            tokens_out_sec = meta.get("tokens_per_second_out", 0)

            row["total_runtime"] = gpu_seconds
            row["tokens_per_second_in"] = tokens_in_sec
            row["tokens_per_second_out"] = tokens_out_sec

            # Million tokens
            # (gpu_seconds * tokens_per_second) / 1e6
            # Note: tokens_per_second is typically avg rate. Total tokens = rate * time.
            total_tokens_in = tokens_in_sec * gpu_seconds
            total_tokens_out = tokens_out_sec * gpu_seconds

            row["million_input_tokens"] = total_tokens_in / 1_000_000.0
            row["million_output_tokens"] = total_tokens_out / 1_000_000.0

            # Cost
            # price per hour * (seconds / 3600)
            cost = standard_price_per_hour * (gpu_seconds / 3600.0)
            row["total_cost"] = cost
            row["cost_per_transcript"] = cost / samples

            data.append(row)

    data.sort(key=lambda x: x["mean_quality"], reverse=True)

    # Create DataFrame
    import pandas as pd

    df = pd.DataFrame(data)

    # Ensure output directory exists
    output_path = "results/df.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved dataframe to {output_path}")
