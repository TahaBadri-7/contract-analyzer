import pandas as pd
df = pd.read_csv("data/CUAD_v1/master_clauses.csv")
for name in ["IVILLAGEINC", "SEPARATEACCOUNT", "PFSFUNDS", "BLACKBOX"]:
    row = df[df["Filename"].str.contains(name)].iloc[0]
    print("\n===", name)
    print("CAP highlighted:", row["Cap On Liability"])
    print("NC highlighted: ", row["Non-Compete"])
    print("EXCLUSIVITY:    ", row["Exclusivity"])