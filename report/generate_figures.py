"""
Generate all figures (PDF) for the research report.
Run: cd report && python generate_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)
CSV_PATH = BASE.parent / "20240110891773837079.csv"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

BEHAVIOR_FEATURES = [
    "duration", "active_span", "page_views", "lecture_item_views",
    "video_views", "forum_posts_count", "forum_comments_count",
]
EN = {
    "duration": "Duration (s)", "active_span": "Active Span (d)",
    "page_views": "Page Views", "lecture_item_views": "Lecture Views",
    "video_views": "Video Views", "forum_posts_count": "Forum Posts",
    "forum_comments_count": "Forum Comments", "grade": "Grade",
}
REGION_MAP = {
    "America": "Americas", "Asia": "Asia", "Europe": "Europe",
    "Africa": "Africa", "Australia": "Oceania", "Pacific": "Oceania",
    "Indian": "Other", "Atlantic": "Other", "Antarctica": "Other",
    "UTC": "Other", "Arctic": "Other",
}


# ── Data loading ──────────────────────────────────────────────────────────
def load():
    raw = pd.read_csv(CSV_PATH)
    if "last_access_time" in raw.columns and "registration_time" in raw.columns:
        raw["active_span"] = (raw["last_access_time"] - raw["registration_time"]) / 86400.0
        raw.loc[raw["active_span"] < 0, "active_span"] = 0
    elif "active_span" not in raw.columns:
        raw["active_span"] = 0
    if "timezone" in raw.columns:
        raw["region"] = raw["timezone"].str.split("/").str[0].map(REGION_MAP).fillna("Other")
    active = raw[raw["grade"] > 0].copy()
    cols = [c for c in BEHAVIOR_FEATURES + ["grade"] if c in active.columns]
    for c in cols:
        active[c] = pd.to_numeric(active[c], errors="coerce")
    active[cols] = active[cols].fillna(0)
    return raw, active


# ── 1. Grade distribution histogram ──────────────────────────────────────
def fig_grade_distribution(active):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(active["grade"], bins=30, color="#6366f1", edgecolor="white", alpha=0.85)
    ax.axvline(active["grade"].mean(), color="#ef4444", ls="--", lw=1.5,
               label=f'Mean {active["grade"].mean():.1f}')
    ax.axvline(active["grade"].median(), color="#f59e0b", ls="--", lw=1.5,
               label=f'Median {active["grade"].median():.1f}')
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    ax.set_title("Grade Distribution")
    ax.legend()
    fig.savefig(FIG_DIR / "grade_distribution.pdf")
    plt.close(fig)


# ── 2. Behavior feature boxplot ──────────────────────────────────────────
def fig_boxplot(active):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(active[cols]),
                          columns=[EN.get(c, c) for c in cols])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = scaled.boxplot(ax=ax, patch_artist=True, return_type="dict",
                        showfliers=False, widths=0.5)
    colors = ["#6366f1", "#8b5cf6", "#ec4899", "#06b6d4",
              "#f59e0b", "#10b981", "#ef4444"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Behavior Feature Distribution (Standardized)")
    ax.set_ylabel("Standardized Value")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(FIG_DIR / "boxplot_features.pdf")
    plt.close(fig)


# ── 3. Correlation heatmap ────────────────────────────────────────────────
def fig_correlation(active):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns] + ["grade"]
    corr = active[cols].corr()
    labels = [EN.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("rg", ["#6366f1", "#ffffff", "#ef4444"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson Correlation Matrix")
    fig.savefig(FIG_DIR / "correlation_heatmap.pdf")
    plt.close(fig)


# ── 4. Scatter plots (top-4 features vs grade) ───────────────────────────
def fig_scatter_top4(active):
    top4 = ["page_views", "duration", "video_views", "lecture_item_views"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    sample = active.sample(min(3000, len(active)), random_state=42)
    for ax, col in zip(axes.flat, top4):
        ax.scatter(sample[col], sample["grade"], s=6, alpha=0.35, color="#6366f1")
        r = active[col].corr(active["grade"])
        ax.set_xlabel(EN.get(col, col))
        ax.set_ylabel("Grade")
        ax.set_title(f"{EN.get(col, col)} vs Grade  (r={r:.3f})")
        ax.grid(alpha=0.2)
    fig.suptitle("Key Behavior Features vs Grade", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_top4.pdf")
    plt.close(fig)


# ── 5. Elbow + silhouette ────────────────────────────────────────────────
def fig_elbow(active):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns]
    X = StandardScaler().fit_transform(active[cols].values)
    ks, inertias, sils = [], [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lab, sample_size=min(3000, len(X))))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, inertias, "o-", color="#6366f1", label="Inertia")
    ax1.set_xlabel("Number of Clusters K")
    ax1.set_ylabel("Inertia", color="#6366f1")
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "s--", color="#ef4444", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="#ef4444")
    ax1.set_title("Elbow Method & Silhouette Score")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.savefig(FIG_DIR / "elbow_silhouette.pdf")
    plt.close(fig)
    return X


# ── 6. PCA cluster scatter ───────────────────────────────────────────────
def fig_cluster_pca(active, X_scaled):
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    idx = np.arange(len(X2))
    if len(idx) > 3000:
        idx = np.random.RandomState(42).choice(idx, 3000, replace=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#6366f1", "#ef4444", "#10b981"]
    for c in range(3):
        mask = labels[idx] == c
        ax.scatter(X2[idx[mask], 0], X2[idx[mask], 1], s=8, alpha=0.5,
                   color=colors[c], label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.set_title("KMeans Clustering - PCA Visualization (K=3)")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.savefig(FIG_DIR / "cluster_pca.pdf")
    plt.close(fig)
    return labels


# ── 7. Cluster radar chart ───────────────────────────────────────────────
def fig_cluster_radar(active, labels):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns]
    df = active.copy()
    df["cluster"] = labels
    means = df.groupby("cluster")[cols].mean()
    normed = (means - means.min()) / (means.max() - means.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    en_labels = [EN.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    colors = ["#6366f1", "#ef4444", "#10b981"]
    for i, row in normed.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, "o-", color=colors[i], label=f"Cluster {i}", lw=1.5)
        ax.fill(angles, vals, color=colors[i], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(en_labels, fontsize=8)
    ax.set_title("Cluster Behavior Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(FIG_DIR / "cluster_radar.pdf")
    plt.close(fig)


# ── 8. Regression: actual vs predicted ───────────────────────────────────
def fig_regression(active):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns]
    X, y = active[cols].values, active["grade"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(Xtr, ytr)
    yp = reg.predict(Xte)

    r2 = r2_score(yte, yp)
    mse = mean_squared_error(yte, yp)
    mae = mean_absolute_error(yte, yp)

    idx = np.arange(len(yte))
    if len(idx) > 500:
        idx = np.random.RandomState(42).choice(idx, 500, replace=False)

    # 8a actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yte[idx], yp[idx], s=12, alpha=0.4, color="#6366f1")
    mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.2)
    ax.set_xlabel("Actual Grade")
    ax.set_ylabel("Predicted Grade")
    ax.set_title(f"Linear Regression: Actual vs Predicted ($R^2$={r2:.4f})")
    ax.grid(alpha=0.15)
    fig.savefig(FIG_DIR / "regression_actual_vs_pred.pdf")
    plt.close(fig)

    # 8b residuals
    residuals = yte - yp
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals[idx], bins=30, color="#8b5cf6", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#ef4444", ls="--")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.savefig(FIG_DIR / "regression_residual.pdf")
    plt.close(fig)

    return r2, mse, mae, reg.coef_, reg.intercept_, cols


# ── 9. Classification model comparison ───────────────────────────────────
def fig_classification(active):
    cols = [c for c in BEHAVIOR_FEATURES if c in active.columns]
    X, y_raw = active[cols].values, active["grade"].values
    threshold = float(np.median(y_raw))
    y = (y_raw >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        threshold = float(np.quantile(y_raw, 0.75))
        y = (y_raw >= threshold).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_sc, Xte_sc = scaler.fit_transform(Xtr), scaler.transform(Xte)

    models = {
        "Logistic Reg.": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=42, early_stopping=True),
    }
    results = {}
    for name, model in models.items():
        use_sc = name == "MLP"
        model.fit(Xtr_sc if use_sc else Xtr, ytr)
        yp = model.predict(Xte_sc if use_sc else Xte)
        results[name] = {
            "acc": accuracy_score(yte, yp),
            "prec": precision_score(yte, yp, zero_division=0),
            "rec": recall_score(yte, yp, zero_division=0),
            "f1": f1_score(yte, yp, zero_division=0),
            "cm": confusion_matrix(yte, yp),
        }

    # 9a metric comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results.keys())
    metrics = ["acc", "prec", "rec", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    w = 0.18
    colors = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"]
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=ml, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIG_DIR / "classification_comparison.pdf")
    plt.close(fig)

    # 9b confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_yticklabels(["Low", "High"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle("Confusion Matrices", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.pdf")
    plt.close(fig)

    return results, threshold


# ── 10. Region comparison ────────────────────────────────────────────────
def fig_region(active):
    region_stats = active.groupby("region")["grade"].agg(["mean", "count"]).reset_index()
    region_stats = region_stats[region_stats["count"] >= 5].sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ["#6366f1", "#8b5cf6", "#ec4899", "#06b6d4", "#f59e0b", "#10b981"]
    bars = ax.bar(region_stats["region"], region_stats["mean"],
                  color=bar_colors[:len(region_stats)], alpha=0.85, edgecolor="white")
    for bar, cnt in zip(bars, region_stats["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"n={cnt}", ha="center", fontsize=8, color="#6b7280")
    ax.set_ylabel("Average Grade")
    ax.set_title("Average Grade by Region")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIG_DIR / "region_grade.pdf")
    plt.close(fig)


# ── 11. Active span bins ────────────────────────────────────────────────
def fig_span(active):
    bins = [0, 7, 30, 60, 90, float("inf")]
    labels = ["<7d", "7-30d", "30-60d", "60-90d", ">90d"]
    ac = active.copy()
    ac["span_bin"] = pd.cut(ac["active_span"], bins=bins, labels=labels, right=False)
    stats = ac.groupby("span_bin", observed=False)["grade"].agg(["mean", "count"]).reset_index()
    stats = stats[stats["count"] > 0]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = range(len(stats))
    ax1.bar(x, stats["mean"], color="#6366f1", alpha=0.8, label="Avg Grade")
    ax1.set_ylabel("Average Grade", color="#6366f1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats["span_bin"])
    ax2 = ax1.twinx()
    ax2.plot(x, stats["count"], "o-", color="#ef4444", label="Count")
    ax2.set_ylabel("Count", color="#ef4444")
    ax1.set_title("Active Span vs Grade")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    fig.savefig(FIG_DIR / "span_grade.pdf")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading data ...")
    raw, active = load()
    print(f"  Total: {len(raw)}, Active: {len(active)}")

    print("1/11 Grade distribution")
    fig_grade_distribution(active)

    print("2/11 Boxplot")
    fig_boxplot(active)

    print("3/11 Correlation heatmap")
    fig_correlation(active)

    print("4/11 Scatter (top-4)")
    fig_scatter_top4(active)

    print("5/11 Elbow + silhouette")
    X_scaled = fig_elbow(active)

    print("6/11 PCA cluster scatter")
    cluster_labels = fig_cluster_pca(active, X_scaled)

    print("7/11 Cluster radar")
    fig_cluster_radar(active, cluster_labels)

    print("8/11 Regression")
    r2, mse, mae, coefs, intercept, feat_names = fig_regression(active)
    print(f"  R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

    print("9/11 Classification")
    cls_results, cls_threshold = fig_classification(active)
    for name, r in cls_results.items():
        print(f"  {name}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")

    print("10/11 Region comparison")
    fig_region(active)

    print("11/11 Active span bins")
    fig_span(active)

    print(f"\nAll figures saved to {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
