import pandas as pd
import numpy as np

INPUT_FILE = "isl_dataset.csv"
OUTPUT_FILE = "isl_normalized_dataset.csv"

df = pd.read_csv(INPUT_FILE)

normalized_rows = []

for _, row in df.iterrows():

    label = row["gesture"]
    features = []

    for hand in range(2):

        hand_data = []

        for point in range(21):
            hand_data.append([
                row[f"h{hand}_x{point}"],
                row[f"h{hand}_y{point}"],
                row[f"h{hand}_z{point}"]
            ])

        points = np.array(hand_data, dtype=float)

        # Keep missing hands as zeros
        if np.allclose(points, 0):
            features.extend([0.0] * 63)
            continue

        # Wrist-relative coordinates
        relative = points - points[0]

        # Scale normalization
        distances = np.linalg.norm(relative, axis=1)
        scale = np.max(distances)

        if scale > 1e-8:
            relative = relative / scale

        features.extend(relative.flatten().tolist())

    normalized_rows.append([label] + features)

columns = ["gesture"]

for hand in range(2):
    for point in range(21):
        columns.extend([
            f"h{hand}_x{point}",
            f"h{hand}_y{point}",
            f"h{hand}_z{point}"
        ])

result = pd.DataFrame(normalized_rows, columns=columns)
result.to_csv(OUTPUT_FILE, index=False)

print("Normalization completed!")
print("Original samples:", len(df))
print("Normalized samples:", len(result))
print("Output file:", OUTPUT_FILE)