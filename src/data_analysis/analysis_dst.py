import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
sns.set_theme(style="whitegrid")
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "dataset" / "raw" / "crypto_combine.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="Crypto", palette="viridis")
plt.title("Number of Records per Cryptocurrency")
plt.ylabel("Count")
plt.xlabel("Cryptocurrency")
plt.show()
