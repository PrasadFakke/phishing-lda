
import pandas as pd

def make_preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            StandardScaler(),
            numeric
        ))

    if categorical:
        from sklearn.preprocessing import OneHotEncoder

        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ))

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def prepare_xy(df, cfg):
    y = (df[cfg["target_column"]] == cfg["positive_label"]).astype(int)
    X = df.drop(
        columns=[cfg["target_column"], *cfg["identifier_columns"]],
        errors="ignore"
    )
    return X, y
