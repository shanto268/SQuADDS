import re

import numpy as np
import pandas as pd
from datasets import load_dataset


def parse_val(val_str):
    """Parse string with units to float. e.g. '50.9um' -> 50.9"""
    if isinstance(val_str, (int, float)):
        return float(val_str)
    if isinstance(val_str, str):
        # Remove units (non-digit/dot/minus)
        match = re.search(r"([-+]?\d*\.?\d+)", val_str)
        if match:
            return float(match.group(1))
    return np.nan


print("Loading dataset...")
dataset = load_dataset("SQuADDS/SQuADDS_DB", "coupler-CapNInterdigitalTee-cap_matrix", split="train")
df = dataset.to_pandas()

# Define features and targets we want
features = ["cap_gap", "cap_width", "finger_count", "finger_length"]
targets = ["bottom_to_bottom", "bottom_to_ground", "ground_to_ground", "top_to_bottom", "top_to_ground", "top_to_top"]

data = []

print("Processing rows...")
for _, row in df.iterrows():
    # Extract features from design_options
    design_opts = row["design"]["design_options"]
    row_data = {}

    for f in features:
        val = design_opts.get(f, np.nan)
        row_data[f] = parse_val(val)

    # Extract targets from sim_results
    sim_res = row["sim_results"]
    for t in targets:
        # sim_results values are already floats usually, but let's be safe
        val = sim_res.get(t, np.nan)
        row_data[t] = val

    data.append(row_data)

df_flat = pd.DataFrame(data)

# Drop rows with NaN if any
df_flat = df_flat.dropna()

output_file = "coupler_capacitance_data.csv"
df_flat.to_csv(output_file, index=False)
print(f"Saved curated dataset to {output_file}")
print(f"Shape: {df_flat.shape}")
print(df_flat.head())
