import os
import sys
import json


if __name__ == "__main__":
    results_files = sys.argv[1:]

    # Load prices
    try:
        with open("input/test_prices.json", "r") as f:
            prices = json.load(f)
            # Default standard price for time-based cost
            standard_price_per_hour = prices.get("gcp_n1_t4", {}).get(
                "dollars_hour_standard", 0
            )
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

            # Categories (fmax for questions, jw_sim for labels)
            # Assumption: Questions end with "?" -> fmax. Others -> jw_sim.

            # Populate from fmax dict (for questions)
            for k, v in item.get("fmax", {}).items():
                if "?" in k:
                    row[k] = v

            # Populate from jw_sim dict (for labels - not questions)
            for k, v in item.get("jw_sim", {}).items():
                if "?" not in k:
                    row[k] = v

            # Metadata & Costs
            model_price_info = prices.get(full_model_name)
            meta = item.get("meta", {})
            samples = meta.get("samples", 446)
            gpu_seconds = meta.get("gpu_seconds", 0)
            tokens_in_sec = meta.get("tokens_per_second_in", 0)
            tokens_out_sec = meta.get("tokens_per_second_out", 0)

            row["total_runtime"] = gpu_seconds
            row["tokens_per_second_in"] = tokens_in_sec
            row["tokens_per_second_out"] = tokens_out_sec
            row["samples"] = samples

            # Million tokens
            # Try to get raw tokens first if available (for exact calculation)
            # otherwise calculate from rate * time
            raw_tokens_in = meta.get("tokens_total_in", tokens_in_sec * gpu_seconds)
            raw_tokens_out = meta.get("tokens_total_out", tokens_out_sec * gpu_seconds)

            million_input_tokens = raw_tokens_in / 1_000_000.0
            million_output_tokens = raw_tokens_out / 1_000_000.0

            row["million_input_tokens"] = million_input_tokens
            row["million_output_tokens"] = million_output_tokens

            # Cost Calculation
            # Logic: Check if model has token-based pricing in test_prices.json
            # If so, use token pricing. Else, fallback to time-based (standard_price_per_hour * time)

            if model_price_info and "dollars_1m_tokens_in" in model_price_info:
                # Token-based pricing
                cost_in = (
                    million_input_tokens * model_price_info["dollars_1m_tokens_in"]
                )
                cost_out = (
                    million_output_tokens * model_price_info["dollars_1m_tokens_out"]
                )
                cost = cost_in + cost_out
            else:
                # Time-based pricing (fallback)
                # If no specific time-based price found for this model, fallback to gcp_n1_t4 standard
                # This assumes non-API models are running on our standard instance

                # We could look for model-specific time pricing if needed, e.g. prices.get(full_model_name, {}).get("dollars_hour_standard")
                # But currently prompt implies using gcp_n1_t4 for others.
                hourly_rate = standard_price_per_hour
                if model_price_info and "dollars_hour_standard" in model_price_info:
                    hourly_rate = model_price_info["dollars_hour_standard"]

                cost = hourly_rate * (gpu_seconds / 3600.0)

            row["total_cost"] = cost
            if samples > 0:
                row["cost_per_transcript"] = cost / samples
            else:
                row["cost_per_transcript"] = 0

            data.append(row)

    # Deduplicate by full_model, keeping the one with the highest samples
    best_entries = {}
    for entry in data:
        model = entry["full_model"]
        curr_samples = entry.get("samples", 0)
        if model not in best_entries or curr_samples > best_entries[model].get("samples", 0):
            best_entries[model] = entry
    data = list(best_entries.values())

    # Sort if mean_quality can be calculated (requires mean_fmax and mean_jw_sim to be non-None)
    # We'll filter out rows where calculation fails for sorting, or treat Nonetype as 0
    def get_quality(r):
        fmax = r.get("mean_fmax") or 0
        jw = r.get("mean_jw_sim") or 0
        return (fmax + jw) / 2

    data.sort(key=get_quality, reverse=True)

    # Create DataFrame
    import pandas as pd

    df = pd.DataFrame(data)

    # Ensure output directory exists
    output_path = "results/df.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved dataframe to {output_path}")

    excel_output = output_path.replace(".csv", ".xlsx")
    df.to_excel(excel_output, index=False)
    print(f"Saved dataframe to {excel_output}")
