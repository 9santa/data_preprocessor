from __future__ import annotations
import pandas as pd

class DataPreprocessor():
    __slots__ = ("df", "removed_cols_", "filled_values_num_", "filled_values_cat_")

    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame(df).copy() # copy so DataPreprocessor class owns the DataFrame
        self.removed_cols_ = []
        self.filled_values_num_ = None
        self.filled_values_cat_ = {}

    def remove_missing(self, 
                       threshold: float=50,          # percentage
                       num_strategy: str = "median", # "median" or "mean"
                       cat_strategy: str = "mode",   # only "mode"
                       return_removed: bool = False  # whether to return removed columns or not
    ):

        percent_missing = self.df.isnull().sum() * 100 / len(self.df)

        columns_removed = percent_missing[percent_missing > threshold].index
        self.removed_cols_ = list(columns_removed)
        columns_to_keep = percent_missing[percent_missing <= threshold].index
        df_removed = self.df.loc[:, columns_removed].copy()
        df_filtered = self.df.loc[:, columns_to_keep].copy()

        # ===== numeric fill =====
        num_cols = df_filtered.select_dtypes(include="number").columns
        if len(num_cols > 0):
            fill_values: pd.Series
            if num_strategy == "median":
                fill_values = df_filtered[num_cols].median(numeric_only=True)
                self.filled_values_num_ = fill_values
            elif num_strategy == "mean":
                fill_values = df_filtered[num_cols].mean(numeric_only=True)
                self.filled_values_num_ = fill_values
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
                    self.filled_values_cat_[col] = df_filtered[col]
                # else: if no values -> leave as NaN


        self.df = df_filtered
        return (df_filtered, df_removed) if return_removed else df_filtered
