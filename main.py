from __future__ import annotations
import pandas as pd

class DataPreprocessor():
    __slots__ = ("df", "removed_cols_", "filled_values_num_", "filled_values_cat_", "onehot_cols_", "normalized_params_",)

    def __init__(self, df: pd.DataFrame):
        self.df = pd.DataFrame(df).copy() # copy so DataPreprocessor class owns the DataFrame
        self.removed_cols_ = []
        self.filled_values_num_ = None
        self.filled_values_cat_ = {}
        self.onehot_cols_ = None
        self.normalized_params_ = {}

    # Removes columns with fraction of missing values above threshold
    # Fills the rest with median / mean / mode
    # Returns modified DataFrame, optionally removed columns
    def remove_missing(self,
                       threshold: float=0.5,         # fraction (0.5 == 50%)
                       num_strategy: str = "median", # "median" or "mean"
                       cat_strategy: str = "mode",   # only "mode"
                       return_removed: bool = False  # whether to return removed columns or not
    ):

        frac_missing = self.df.isna().mean() # fraction missing per column

        columns_removed = frac_missing[frac_missing > threshold].index
        self.removed_cols_ = list(columns_removed)
        columns_to_keep = frac_missing[frac_missing <= threshold].index
        df_removed = self.df.loc[:, columns_removed].copy()
        df_filtered = self.df.loc[:, columns_to_keep].copy()

        # ===== numeric fill =====
        num_cols = df_filtered.select_dtypes(include="number").columns
        if len(num_cols) > 0:
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
                    self.filled_values_cat_[col] = mode_vals.iloc[0]
                # else: if no values -> leave as NaN


        self.df = df_filtered
        return (df_filtered, df_removed) if return_removed else df_filtered

    # One-hot encoding of categorical columns
    # Returns modified DataFrame
    def encode_categorical(self):
        cat_cols = self.df.select_dtypes(include=["string", "object", "category"]).columns.to_list()
        self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=False) # each variable is converted into 0s/1s
        self.onehot_cols_ = list(self.df.columns) # save
        return self.df

    # Normalizes numeric columns with 'minmax' or 'std'
    # Returns modified DataFrame
    def normalize_numeric(self, method: str="minmax"):
        num_cols = self.df.select_dtypes(include="number").columns.to_list()

        if method == "minmax": # minmax
            mn = self.df[num_cols].min()
            mx = self.df[num_cols].max()
            self.normalized_params_ = {"method": "minmax", "min": mn, "max": mx} # save

            # Handle division by 0 carefully
            denom = mx - mn
            zero_denom_cols = denom[denom == 0].index

            denom_safe = denom.replace(0, 1)
            self.df[num_cols] = (self.df[num_cols] - mn) / denom_safe
            self.df.loc[:, zero_denom_cols] = 0

        elif method == "std": # std
            mean = self.df[num_cols].mean()
            std = self.df[num_cols].std()
            self.normalized_params_ = {"method": "std", "mean": mean, "std": std} # save

            # Handle division by 0 carefully
            zero_std_cols = std[std == 0].index
            std_safe = std.replace(0, 1)

            self.df[num_cols] = (self.df[num_cols] - mean) / std_safe
            self.df.loc[:, zero_std_cols] = 0

        else:
            raise ValueError("method must be 'minmax' or 'std'")

        return self.df

    # Full pipeline, applies all transformations
    # Returns modified DataFrame
    def fit_transform(self, threshold: int=50, num_strategy: str="median", method: str="minmax"):
        self.remove_missing(threshold=threshold, num_strategy=num_strategy, cat_strategy="mode")
        self.encode_categorical()
        self.normalize_numeric(method=method)
        return self.df
