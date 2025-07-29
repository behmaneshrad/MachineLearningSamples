#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wine_quality_pipeline.py
=======================

کامل‌ترین پایپ‌لاین پیش‌بینی کیفیت شراب (red & white)
با استفاده از داده‌های UCI Wine-Quality :
    - بارگذاری داده‌ها
    - آماده‌سازی
    - بررسی آماری و تصویری
    - مدل‌های رگرسیون و رده‌بندی
    - ارزیابی و اهمیت ویژگی‌ها

اجرا:
    python wine_quality_pipeline.py

نیازمندی‌ها:
    pip install pandas scikit-learn matplotlib seaborn
"""

# -------------------- 0. Imports --------------------
import os
import warnings
import json
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             classification_report, roc_auc_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.3f}".format)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)

# -------------------- 1. Paths --------------------
DATA_DIR = Path("dataset")
RED_FILE   = DATA_DIR / "winequality-red.csv"
WHITE_FILE = DATA_DIR / "winequality-white.csv"

# -------------------- 2. Functions --------------------
def load_data():
    """بارگذاری دو CSV و ترکیب آن‌ها با ستون نوع شراب"""
    red   = pd.read_csv(RED_FILE,   sep=";")
    white = pd.read_csv(WHITE_FILE, sep=";")

    red["type"]   = "red"
    white["type"] = "white"
    df = pd.concat([red, white], ignore_index=True)
    print(f"Loaded {len(red)} red + {len(white)} white = {len(df)} rows")
    return df

def display_df_info(df, name="DataFrame"):
    """نمایش سریع اطلاعات داده"""
    print(f"\n=== {name} Info ===")
    print("Shape :", df.shape)
    print("Head  :\n", df.head(3))
    print("Describe:\n", df.describe().T.iloc[:, :4])
    print("Missing:\n", df.isna().sum())

def plot_quality_distribution(df):
    """نمودار توزیع کیفیت برای هر نوع شراب"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x="quality", data=df[df.type == "red"],   ax=ax[0], color="crimson")
    sns.countplot(x="quality", data=df[df.type == "white"], ax=ax[1], color="gold")
    ax[0].set_title("Red – quality")
    ax[1].set_title("White – quality")
    plt.suptitle("Distribution of wine quality scores")
    plt.tight_layout()
    plt.show()

def prepare_data(df):
    """جدا کردن ویژگی‌ها و هدف + اسکیل کردن"""
    X = df.drop(columns=["quality", "type"])
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["type"]
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nTrain / Test split finished")
    print("Train:", X_train_scaled.shape, "Test:", X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

def regression_results(model, X_test, y_test, model_name):
    """ارزیابی رگرسیون و نمایش نمودار"""
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2   = r2_score(y_test, pred)
    print(f"\n=== {model_name} ===")
    print("RMSE:", rmse, "   R2:", r2)

    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=y_test, y=pred)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--")
    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title(f"{model_name} – Predicted vs True")
    plt.tight_layout()
    plt.show()
    return rmse, r2

def plot_coefficients(models, names, feature_names):
    """نمودار مقایسه ضرایب مدل‌ها"""
    coef_df = pd.DataFrame(index=feature_names)
    for mdl, name in zip(models, names):
        coef_df[name] = mdl.coef_
    coef_df = coef_df.sort_values(names[0])
    coef_df.plot(kind="barh", figsize=(8, 8))
    plt.title("Coefficient comparison")
    plt.axvline(0, color="black")
    plt.tight_layout()
    plt.show()

def plot_permutation_importance(model, X_test, y_test, feature_names):
    """اهمیت ویژگی‌ها با permutation"""
    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=30, random_state=42)
    imp = pd.Series(perm.importances_mean, index=feature_names)\
            .sort_values(ascending=False)
    print("\n=== Permutation importance (top 10)")
    print(imp.head(10))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=imp.head(10), y=imp.head(10).index)
    plt.title("Top 10 important features")
    plt.tight_layout()
    plt.show()

def classification_view(X_train, X_test, y_train, y_test, feature_names):
    """رده‌بندی دودویی quality >= 7"""
    y_train_cls = (y_train >= 7).astype(int)
    y_test_cls  = (y_test  >= 7).astype(int)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train_cls)

    pred_cls = clf.predict(X_test)
    print("\n=== Logistic Regression (quality >= 7)")
    print(classification_report(y_test_cls, pred_cls))
    auc = roc_auc_score(y_test_cls, clf.decision_function(X_test))
    print("ROC-AUC:", auc)

# -------------------- 3. Main Pipeline --------------------
def main():
    df = load_data()
    display_df_info(df, "Wine Quality Combined")
    plot_quality_distribution(df)

    X_train, X_test, y_train, y_test, feat_names, scaler = prepare_data(df)

    # --- Linear Regression ---
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    regression_results(lin, X_test, y_test, "Linear Regression")

    # --- Ridge & Lasso ---
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
    ridge.fit(X_train, y_train)
    regression_results(ridge, X_test, y_test, "RidgeCV")

    lasso = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000)
    lasso.fit(X_train, y_train)
    regression_results(lasso, X_test, y_test, "LassoCV")

    plot_coefficients([lin, ridge, lasso],
                      ["Linear", "Ridge", "Lasso"],
                      feat_names)

    plot_permutation_importance(lin, X_test, y_test, feat_names)

    # --- Classification (optional) ---
    classification_view(X_train, X_test, y_train, y_test, feat_names)

if __name__ == "__main__":
    main()