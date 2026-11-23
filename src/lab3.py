"""Спільний код лабораторної №3: дані, поділ, моделі, перехресна валідація."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "auto-mpg.data"
SEED = 42
N_SPLITS, N_REPEATS = 5, 10
COLS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name"]
BRAND_FIX = {"chevroelt": "chevrolet", "chevy": "chevrolet", "toyouta": "toyota",
             "maxda": "mazda", "vokswagen": "volkswagen", "vw": "volkswagen",
             "mercedes-benz": "mercedes", "capri": "mercury",
             "hi": "international harvester"}


def load_data():
    """Читає сирий файл UCI. Пропуски НЕ заповнює: це робиться всередині розбиттів."""
    df = pd.read_csv(RAW, sep=r"\s+", names=COLS, na_values="?", quotechar='"')
    df["origin"] = df["origin"].map({1: "США", 2: "Європа", 3: "Японія"})
    parts = df["car_name"].str.split(n=1, expand=True)
    brand = parts[0].replace(BRAND_FIX)
    # об'єкт = назва моделі: один автомобіль може повторюватися в різні роки
    df["model_id"] = (brand + " " + parts[1].fillna("")).str.strip()
    df["w100"] = df["weight"] * 0.45359 / 100      # маса, сотні кг
    df["hp10"] = df["horsepower"] / 10             # потужність, десятки к. с.
    df["yr"] = df["model_year"] - 76               # роки від 1976
    df["displ"] = df["displacement"] * 0.016387    # об'єм, л
    return df


def split(df):
    """80/20 за об'єктами (назвами моделей), фіксований seed."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["model_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


class LogTarget(RegressorMixin, BaseEstimator):
    """Регресія для ln(mpg) з поправкою Дуана (smearing) при зворотному перетворенні."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        ly = np.log(y)
        self.regressor_ = clone(self.regressor).fit(X, ly)
        self.smear_ = float(np.mean(np.exp(ly - self.regressor_.predict(X))))
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.smear_


def _prep(num_steps, num):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), *num_steps), num),
        ("cat", OneHotEncoder(categories=[["США", "Європа", "Японія"]], drop="first"), ["origin"])])


def _quad(X):
    """Додає (w-13)²; центр 13 = 1300 кг заданий наперед."""
    X = np.asarray(X, float)
    w = X[:, 0] - 13
    return np.column_stack([X, w ** 2])


def _quad_int(X):
    X = np.asarray(X, float)
    w, yr = X[:, 0] - 13, X[:, 2]
    return np.column_stack([X, w ** 2, w * yr])


def _log2(X):
    X = np.asarray(X, float)
    return np.column_stack([np.log(X[:, 0]), np.log(X[:, 1]), X[:, 2]])


BASE = ["w100", "hp10", "yr"]
ALL = ["w100", "hp10", "yr", "displ", "cylinders", "acceleration"]


def make_models():
    ft = FunctionTransformer
    return {
        "M0 константа": DummyRegressor(strategy="mean"),
        "M1 лінійна": make_pipeline(_prep([], BASE), LinearRegression()),
        "M2a + маса²": make_pipeline(_prep([ft(_quad)], BASE), LinearRegression()),
        "M2b + маса² + маса×рік": make_pipeline(_prep([ft(_quad_int)], BASE), LinearRegression()),
        "M2c log-log": LogTarget(make_pipeline(_prep([ft(_log2)], BASE), LinearRegression())),
        "M2d ridge, усі ознаки": make_pipeline(
            _prep([], ALL), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "M3 випадковий ліс": make_pipeline(
            _prep([], ALL), RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    }


def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def cross_validate(train, models):
    """Повторна групова CV: ті самі розбиття для всіх моделей."""
    rows = []
    for r in range(N_REPEATS):
        cv = GroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r)
        for f, (a, b) in enumerate(cv.split(train, groups=train["model_id"])):
            for n, m in models.items():
                p = clone(m).fit(train.iloc[a], train["mpg"].iloc[a]).predict(train.iloc[b])
                rows.append({"model": n, "repeat": r, "fold": f, **metrics(train["mpg"].iloc[b], p)})
    return pd.DataFrame(rows)
