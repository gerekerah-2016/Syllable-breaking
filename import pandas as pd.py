import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data you processed
CSV_PATH = "geez_training_data.csv"
df = pd.read_csv(CSV_PATH)

# 2. Set up the visual style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 3. Create the Histogram
sns.histplot(df['length'], kde=True, color='teal', bins=50)

# 4. Add statistical lines
mean_len = df['length'].mean()
p95_len = df['length'].quantile(0.95) # The length that covers 95% of data

plt.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.1f}')
plt.axvline(p95_len, color='orange', linestyle='--', label=f'95th Percentile: {p95_len:.1f}')

plt.title('Distribution of Tokenized Ge\'ez Sequence Lengths', fontsize=15)
plt.xlabel('Number of Tokens', fontsize=12)
plt.ylabel('Number of Verses', fontsize=12)
plt.legend()

plt.show()

print(f"95% of your sequences are shorter than {p95_len:.1f} tokens.")