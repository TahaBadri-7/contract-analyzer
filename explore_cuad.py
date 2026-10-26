import pandas as pd

df = pd.read_csv("data/CUAD_v1/master_clauses.csv")

print(df.shape)                      # (rows, columns)
print(list(df.columns)[:12])         # first dozen column names

cols = ["Filename", "Governing Law-Answer", "Cap On Liability-Answer", "Non-Compete-Answer"]
print(df[cols].head(10).to_string())

print(df["Cap On Liability-Answer"].value_counts(dropna=False))
print(df["Non-Compete-Answer"].value_counts(dropna=False))
