# Dataset profile

Inspected local file: `data/phishing.csv` (855,304 bytes). It contains 11,054 rows and 32 columns. All 32 columns are `int64`; the 30 named website/URL indicators, `Index`, and `class` are structured extracted features, with no raw URL field.

| Audit item | Finding |
|---|---|
| Target | `class` |
| Labels | `-1` phishing (4,897); `1` legitimate (6,157) |
| Positive label for this project | `-1`, remapped to 1 |
| Missing values | 0 in every column |
| Exact duplicate rows | 0 |
| Constant columns | None |
| Identifier | `Index`, 11,054 unique values; excluded from predictors |
| Predictor columns | 30, integer coded |
| Class balance | 44.3% phishing, 55.7% legitimate |

The schema matches the 30-indicator Phishing Websites dataset. Existing row order is sequential in `Index`; no predefined split or entity key is present. Random stratified split is used as specified, but the dataset does not expose domains/URLs to support grouping or duplicate-entity leakage checks beyond exact-row duplicate detection. Feature semantics cover URL structure, domain and page behaviors, rank and external reputation signals; this is not raw-URL lexical classification. Labels and counts are corroborated by the dataset literature [1].

All predictor columns are variable. Duplicate rows across complete records are absent. All fields are already numeric ordinal/binary codes, so no categorical encoding is needed for this file. Numeric imputation/scaling is included in the fitted pipeline; scaling is fit only inside training folds. No resampling is applied.
