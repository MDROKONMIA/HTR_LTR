# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # === Step 1: Load your CSV file ===
# df = pd.read_csv('common_subdirectories.csv')
#
# # Apply leading zeros (7 digits) to all values
# df = df.applymap(lambda x: f"{int(x):07d}")
#
# # === Step 2: Split data into 50% train and 50% test ===
# train_df, test_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)
#
# # === Step 3: Save the two splits into CSV files ===
# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)
#
# # === Step 4: Convert the splits into JSON files ===
# train_df.to_json('train.json', orient='records', lines=True)
# test_df.to_json('test.json', orient='records', lines=True)
#
# print("✅ Files created successfully: 'train.csv', 'test.csv', 'train.json', and 'test.json'")
#


import csv
import json

# Input and output file paths
csv_file = "train.csv"   # your CSV file name
json_file = "test.json"  # output JSON file

# Read CSV file
with open(csv_file, "r", newline="") as f:
    reader = csv.reader(f)
    data = []

    # Flatten all rows (handles single-column CSV)
    for row in reader:
        for item in row:
            item = item.strip()
            if item:  # ignore empty cells
                data.append(item)

# Write to JSON file
with open(json_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ Successfully converted {csv_file} to {json_file}")
