import pandas as pd
from sklearn.metrics import cohen_kappa_score

###############################################################################
# 1. Helper ─ read one file and trim to Hassan’s sample
###############################################################################
def load_and_trim(path, filetype="json"):
    """
    Load a Unity-Catalog or Crawl4AI annotation file and keep only
    the rows Hassan annotated.  Returns a DataFrame with two columns:
    'Alor_label' and 'Hassan_label'.
    """
    if filetype == "json":
        df = pd.read_json(path)          # orient="records" is default
    else:                                # CSV fallback
        df = pd.read_csv(path)
        
    # Treat empty strings and explicit NaN the same
    mask = df["Hassan_label"].notna() & (df["Hassan_label"] != "")
    trimmed = df.loc[mask, ["Alor_label", "Hassan_label"]]
    return trimmed

###############################################################################
# 2. Load the two parts of your dataset
###############################################################################
unity   = load_and_trim("src/utils/doc_code_pairs_unity_catalog.json")      # or .csv
crawl4  = load_and_trim("src/utils/doc_code_pairs_crawl4ai.json")           # or .csv

###############################################################################
# 3. Compute κ for each file individually (nice for the appendix)
###############################################################################
def print_kappa(label, df):
    κ = cohen_kappa_score(df["Alor_label"], df["Hassan_label"])
    print(f"{label:14}  n = {len(df):4d}   κ = {κ:.3f}")

print_kappa("Unity Catalog", unity)
print_kappa("Crawl4AI",      crawl4)

###############################################################################
# 4. Pool both samples and compute the overall κ
###############################################################################
pooled = pd.concat([unity, crawl4], ignore_index=True)
κ_pooled = cohen_kappa_score(pooled["Alor_label"], pooled["Hassan_label"])
print(f"\nPooled Sample   n = {len(pooled):4d}   κ = {κ_pooled:.3f}")
