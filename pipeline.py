import gc
import glob
import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


SESSIONS_PREFIX = "ga_sessions"
HITS_PREFIX = "ga_hits"

TARGET_ACTIONS = [
    "sub_car_claim_click",
    "sub_car_claim_submit_click",
    "sub_open_dialog_click",
    "sub_custom_question_submit_click",
    "sub_call_number_click",
    "sub_callback_submit_click",
    "sub_submit_success",
    "sub_car_request_submit_click"]

DATA_READERS = [(".parquet", pd.read_parquet), (".pkl", pd.read_pickle), (".csv", pd.read_csv)]

def read_data_file(prefix):
    for extention, reader in DATA_READERS:
        matches = sorted(glob.glob(f"{prefix}*{extention}"))
        if matches:
            path = matches[0]
            if len(matches) > 1:
                print(f"Внимание: найдено несколько файлов {matches}, "
                      f"используется {path}")
            print(f"Найден файл: {path}")
            return reader(path)

    tried = ",".join(f"{prefix}*{ext}" for ext, _ in DATA_READERS)
    raise FileNotFoundError(f"Не найден ни один файл по шаблонам: {tried}."
                            f"Положите файлы данных рядом с проектом")

def load_data():
    print("Загрузка GA Hits...")
    hitc = read_data_file(HITS_PREFIX)
    hits = hitc[["session_id", "event_action"]].copy()

    del hitc
    gc.collect()
    print("GA Hits загружены.")

    hits["is_target"] = (hits["event_action"].isin(TARGET_ACTIONS).astype("int8"))
    target_df = (hits.groupby("session_id", as_index=False)["is_target"].max().rename(columns={"is_target": "target"}))

    del hits
    gc.collect()
    print("Target сформирован.")

    print("Загрузка GA Sessions...")
    sessions = read_data_file(SESSIONS_PREFIX)
    print(f"GA Sessions: {sessions.shape}")
    df = sessions.merge(target_df, on="session_id", how="left")

    del sessions
    del target_df
    gc.collect()

    df["target"] = (df["target"].fillna(0).astype("int8"))
    print(f"Итоговый датасет: {df.shape}")
    print(f"Доля target=1: {df['target'].mean():.4f}")

    return df

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["has_keyword"] = (X["utm_keyword"].notna().astype("int8"))
        X["is_russia"] = (X["geo_country"] == "Russia").astype("int8")
        X["is_presence_city"] = (X["geo_city"].isin(["Moscow", "Saint Petersburg"]).astype("int8"))
        X["visit_date"] = pd.to_datetime(X["visit_date"], errors="coerce")
        X["visit_weekday"] = (X["visit_date"].dt.dayofweek)
        X["visit_hour"] = (pd.to_datetime(X["visit_time"], format="%H:%M:%S", errors="coerce").dt.hour)

        resolution = (X["device_screen_resolution"].astype("string"))
        bad_resolution = resolution.isin(["0x0", "(not set)", "1x1", ""])
        resolution = resolution.mask(bad_resolution, pd.NA)
        split_resolution = resolution.str.split("x", expand=True)
        X["screen_width"] = pd.to_numeric(split_resolution[0], errors="coerce")
        X["screen_height"] = pd.to_numeric(split_resolution[1], errors="coerce")
        X["device_brand"] = (X["device_brand"].astype("string"))
        X["device_brand"] = (X["device_brand"].replace("", pd.NA))

        mask_desktop = (X["device_category"] == "desktop")
        mask_other_missing = (X["device_brand"].isna() & ~mask_desktop)
        X.loc[mask_desktop, "device_brand"] = (X.loc[mask_desktop, "device_brand"].fillna("desktop_no_brand"))
        X.loc[mask_other_missing, "device_brand"] = ("brand_unknown")

        mask_desktop_ns = ((X["device_category"] == "desktop") & (X["device_brand"] == "(not set)"))
        mask_other_ns = ((X["device_category"] != "desktop") & (X["device_brand"] == "(not set)"))

        X.loc[mask_desktop_ns, "device_brand"] = ("desktop_no_brand")
        X.loc[mask_other_ns, "device_brand"] = ("brand_unknown")

        return X

class TopNGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, top_n):
        self.top_n = top_n

    def fit(self, X, y=None):
        X = X.copy()
        self.top_values_ = {}
        for column, n in self.top_n.items():
            values = (X[column].astype("string").value_counts().head(n).index.tolist())
            self.top_values_[column] = set(values)

        return self

    def transform(self, X):
        X = X.copy()
        for column in self.top_n:
            top_values = self.top_values_[column]
            grouped_column = f"{column}_grouped"
            values = (X[column].astype("string"))
            X[grouped_column] = (values.where(values.isin(top_values), "other"))

        return X

class DataCleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["utm_source"] = (X["utm_source"].fillna("unknown_source"))
        X["device_os"] = (X["device_os"].fillna("os_not_detected"))
        X["geo_city"] = (X["geo_city"].fillna("unknown_city"))
        X["utm_campaign"] = (X["utm_campaign"].fillna("no_campaign"))
        X["utm_adcontent"] = (X["utm_adcontent"].fillna("no_adcontent"))

        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "target" in X.columns:
            X = X.drop(columns=["target"])
        drop_cols = ["session_id", "client_id"]
        X = X.drop(columns=drop_cols, errors="ignore")
        drop_cols = ["utm_keyword", "geo_country", "geo_city", "utm_source", "utm_campaign", "device_brand", "utm_adcontent", "visit_time", "device_screen_resolution", "visit_date"]
        X = X.drop(columns=drop_cols, errors="ignore")

        return X

class ScreenHeightClipper(BaseEstimator, TransformerMixin):
    def __init__(self, upper=2160):
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "screen_height" in X.columns:
            X["screen_height"] = (X["screen_height"].clip(upper=self.upper))

        return X

def build_pipeline():
    numeric_features = ["visit_number", "has_keyword", "is_russia", "is_presence_city", "visit_weekday", "visit_hour", "screen_width", "screen_height"]
    categorical_features = ["utm_medium", "device_category", "device_os", "device_browser", "geo_city_grouped", "utm_source_grouped", "utm_campaign_grouped", "device_brand_grouped", "utm_adcontent_grouped"]
    top_n = {"geo_city": 20, "utm_source": 10, "utm_campaign": 15, "device_brand": 10, "utm_adcontent": 10}

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_features), ("categorical", categorical_pipeline, categorical_features)], remainder="drop")
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=10, min_samples_leaf=20)
    pipeline = Pipeline(
        steps=[("feature_engineering", FeatureEngineering()),
               ("data_cleaning", DataCleaning()),
               ("top_n_grouping", TopNGroupTransformer(top_n=top_n)),
               ("screen_height_clipping", ScreenHeightClipper(upper=2160)),
               ("feature_selection", FeatureSelector()),
               ("preprocessor", preprocessor),
               ("model", model)])

    return pipeline

