import pandas as pd
class Preprocessor:
    def __init__(self, drop_cols=None, cat_cols=None):
        self.drop_cols = drop_cols or []
        self.cat_cols = cat_cols or []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop(columns=self.drop_cols, inplace=True, errors="ignore")
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df