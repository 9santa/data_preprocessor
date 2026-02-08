from __future__ import annotations
import pandas as pd

class DataPreprocessor():
    __slots__ = ("df",)

    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame(df).copy()

    def remove_missing(self, 
                       threshold: float=50,
                       num_strategy: str = "median",
                       cat_strategy: str = "mode",
                       return_removed: bool = False
    ):

        percent_missing = self.df.isnull().sum() * 100 / len(self.df)

        columns_removed = percent_missing[percent_missing > threshold].index
        columns_to_keep = percent_missing[percent_missing <= threshold].index
        df_removed = self.df.loc[:, columns_removed].copy()
        df_filtered = self.df.loc[:, columns_to_keep].copy()

        # ===== numeric fill =====
        num_cols = df_filtered.select_dtypes(include="number").columns
        if len(num_cols > 0):
            fill_values: pd.Series
            if num_strategy == "median":
                fill_values = df_filtered[num_cols].median(numeric_only=True)
            elif num_strategy == "mean":
                fill_values = df_filtered[num_cols].mean(numeric_only=True)
            else:
                raise ValueError("numerical fill strategy must be 'median' or 'mean'")

            df_filtered.loc[:, num_cols] = df_filtered.loc[:, num_cols].fillna(fill_values)


        # ===== categorical / text fill =====
        non_num_cols = df_filtered.columns.difference(num_cols)
        if cat_strategy != "mode":
            raise ValueError("categorical fill strategy must be 'mode'")

        for col in non_num_cols:
            if df_filtered[col].isna().any():
                mode_vals = df_filtered[col].mode(dropna=True)
                if not mode_vals.empty:
                    df_filtered[col] = df_filtered[col].fillna(mode_vals.iloc[0])
                # else: if no values -> leave as NaN


        self.df = df_filtered
        return (df_filtered, df_removed) if return_removed else df_filtered
