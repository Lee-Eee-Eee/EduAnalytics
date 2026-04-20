"""
在线学习行为数据分析可视化平台
Flask 后端 —— 数据预处理、统计分析、机器学习建模、AI 聊天代理

v4: 拖拽上传 CSV、API 配置浏览器端持久化
"""

import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests as http_requests
from flask import Flask, Response, render_template, request, jsonify
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

# ---------------------------------------------------------------------------
# 列信息映射
# ---------------------------------------------------------------------------
COLUMN_CN = {
    "duration": "在线时长(秒)",
    "video_views": "视频观看次数",
    "page_views": "页面浏览次数",
    "lecture_item_views": "课件浏览次数",
    "forum_posts_count": "发帖数",
    "forum_comments_count": "评论数",
    "active_span": "活跃天数",
    "grade": "最终成绩",
    "region": "地区",
    "deleted": "账户状态",
}

BEHAVIOR_FEATURES = [
    "duration",
    "active_span",
    "page_views",
    "lecture_item_views",
    "video_views",
    "forum_posts_count",
    "forum_comments_count",
]

REGION_MAP = {
    "America": "美洲", "Asia": "亚洲", "Europe": "欧洲", "Africa": "非洲",
    "Australia": "大洋洲", "Pacific": "大洋洲",
    "Indian": "其他", "Atlantic": "其他", "Antarctica": "其他", "UTC": "其他", "Arctic": "其他",
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def safe(obj):
    if isinstance(obj, dict):
        return {safe(k): safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return safe(obj.tolist())
    if isinstance(obj, (pd.Series, pd.Index)):
        return safe(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def three_rates(series):
    """计算及格率(>=60)、优秀率(>=80)、拔尖率(>=90)"""
    n = len(series)
    if n == 0:
        return 0, 0, 0
    return (
        float((series >= 60).sum() / n),
        float((series >= 80).sum() / n),
        float((series >= 90).sum() / n),
    )


# ---------------------------------------------------------------------------
# 数据加载与清洗
# ---------------------------------------------------------------------------

def prepare_dataframe(raw):
    # 派生特征
    if "last_access_time" in raw.columns and "registration_time" in raw.columns:
        raw["active_span"] = (raw["last_access_time"] - raw["registration_time"]) / 86400.0
        raw.loc[raw["active_span"] < 0, "active_span"] = 0
    elif "active_span" not in raw.columns:
        raw["active_span"] = 0

    if "timezone" in raw.columns:
        raw["region"] = raw["timezone"].str.split("/").str[0].map(REGION_MAP).fillna("其他")
    elif "region" not in raw.columns:
        raw["region"] = "未知"

    total_n = len(raw)
    zero_n = int((raw["grade"] == 0).sum()) if "grade" in raw.columns else 0

    active = raw[raw["grade"] > 0].copy() if "grade" in raw.columns else raw.copy()

    num_cols = [c for c in BEHAVIOR_FEATURES + ["grade"] if c in active.columns]
    for col in num_cols:
        active[col] = pd.to_numeric(active[col], errors="coerce")
    active[num_cols] = active[num_cols].fillna(0)

    return raw, active, total_n, zero_n


# ---------------------------------------------------------------------------
# 分析函数
# ---------------------------------------------------------------------------

def compute_overview(raw, active, total_n, zero_n):
    n = len(active)
    g = active["grade"]
    avg_grade = g.mean()
    median_grade = g.median()
    std_grade = g.std()
    pass_rate, excellent_rate, top_rate = three_rates(g)
    dropout_rate = zero_n / total_n if total_n else 0

    counts, bin_edges = np.histogram(g, bins=25)
    grade_hist = {
        "bins": [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))],
        "counts": counts.tolist(),
    }

    feat_stats = []
    for col in BEHAVIOR_FEATURES:
        if col in active.columns:
            s = active[col]
            feat_stats.append({
                "col": col, "cn": COLUMN_CN.get(col, col),
                "mean": s.mean(), "median": s.median(), "std": s.std(),
                "min": s.min(), "max": s.max(),
                "q1": s.quantile(0.25), "q3": s.quantile(0.75),
            })

    boxplot_data = []
    for col in BEHAVIOR_FEATURES:
        if col in active.columns:
            s = active[col]
            boxplot_data.append({
                "name": COLUMN_CN.get(col, col), "col": col,
                "min": float(s.min()), "q1": float(s.quantile(0.25)),
                "median": float(s.median()), "q3": float(s.quantile(0.75)),
                "max": float(s.quantile(0.95)),
            })

    return {
        "total_registered": total_n, "n_active": n, "n_zero": zero_n,
        "dropout_rate": dropout_rate,
        "avg_grade": avg_grade, "median_grade": median_grade, "std_grade": std_grade,
        "pass_rate": pass_rate, "excellent_rate": excellent_rate, "top_rate": top_rate,
        "grade_hist": grade_hist, "feat_stats": feat_stats, "boxplot": boxplot_data,
    }


def compute_correlation(df):
    cols = [c for c in BEHAVIOR_FEATURES if c in df.columns] + ["grade"]
    sub = df[cols].copy()
    corr_matrix = sub.corr()
    grade_corr = corr_matrix["grade"].drop("grade").sort_values(ascending=False)
    return {
        "columns": [COLUMN_CN.get(c, c) for c in cols],
        "columns_en": cols,
        "matrix": corr_matrix.values.tolist(),
        "grade_corr": [
            {"feature": COLUMN_CN.get(k, k), "feature_en": k, "value": v}
            for k, v in grade_corr.items()
        ],
    }


def compute_scatter(df):
    cols = [c for c in BEHAVIOR_FEATURES if c in df.columns] + ["grade"]
    sub = df[cols].copy()
    if len(sub) > 5000:
        sub = sub.sample(5000, random_state=42)
    data = {col: sub[col].tolist() for col in cols}
    corrs = {}
    for col in BEHAVIOR_FEATURES:
        if col in df.columns:
            corrs[col] = float(df[col].corr(df["grade"]))
    return {"data": data, "correlations": corrs, "n_sampled": len(sub)}


def compute_group_analysis(raw, active):
    region_stats = []
    for region in ["美洲", "亚洲", "欧洲", "大洋洲", "非洲", "其他"]:
        g = active[active["region"] == region]
        if len(g) == 0:
            continue
        pr, er, tr = three_rates(g["grade"])
        region_stats.append({
            "region": region, "count": len(g),
            "avg_grade": float(g["grade"].mean()),
            "median_grade": float(g["grade"].median()),
            "std_grade": float(g["grade"].std()) if len(g) > 1 else 0,
            "pass_rate": pr, "excellent_rate": er, "top_rate": tr,
        })

    region_dropout = []
    for region in ["美洲", "亚洲", "欧洲", "大洋洲", "非洲", "其他"]:
        total = raw[raw["region"] == region]
        active_r = total[total["grade"] > 0]
        if len(total) == 0:
            continue
        region_dropout.append({
            "region": region, "total": len(total), "active": len(active_r),
            "dropout_rate": float(1 - len(active_r) / len(total)),
        })

    deleted_stats = []
    for d, label in [(0, "保留账户"), (1, "已删除账户")]:
        g = active[active["deleted"] == d] if "deleted" in active.columns else active.iloc[0:0]
        if len(g) == 0:
            continue
        pr, er, tr = three_rates(g["grade"])
        deleted_stats.append({
            "status": label, "deleted": d, "count": len(g),
            "avg_grade": float(g["grade"].mean()),
            "median_grade": float(g["grade"].median()),
            "pass_rate": pr, "excellent_rate": er, "top_rate": tr,
        })

    bins = [0, 7, 30, 60, 90, float("inf")]
    labels = ["<7天", "7-30天", "30-60天", "60-90天", ">90天"]
    active_c = active.copy()
    active_c["span_bin"] = pd.cut(active_c["active_span"], bins=bins, labels=labels, right=False)
    span_stats = []
    for label in labels:
        g = active_c[active_c["span_bin"] == label]
        if len(g) == 0:
            continue
        pr, er, tr = three_rates(g["grade"])
        span_stats.append({
            "bin": label, "count": len(g),
            "avg_grade": float(g["grade"].mean()),
            "median_grade": float(g["grade"].median()),
            "pass_rate": pr, "excellent_rate": er, "top_rate": tr,
        })

    return {
        "region_stats": region_stats,
        "region_dropout": region_dropout,
        "deleted_stats": deleted_stats,
        "span_stats": span_stats,
    }


def compute_clustering(df):
    use_cols = [c for c in BEHAVIOR_FEATURES if c in df.columns]
    X = df[use_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, lab, sample_size=min(3000, len(X_scaled)))
        elbow.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels3 = km3.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.tolist()

    idx = np.arange(len(X_pca))
    if len(idx) > 3000:
        idx = np.random.RandomState(42).choice(idx, 3000, replace=False)

    pca_points = [
        {"x": float(X_pca[i, 0]), "y": float(X_pca[i, 1]),
         "cluster": int(labels3[i]), "grade": float(df["grade"].iloc[i])}
        for i in idx
    ]

    df_c = df.copy()
    df_c["cluster"] = labels3
    cluster_stats = []
    for c_id in sorted(df_c["cluster"].unique()):
        g = df_c[df_c["cluster"] == c_id]
        stat = {"cluster": int(c_id), "count": len(g),
                "avg_grade": float(g["grade"].mean()), "median_grade": float(g["grade"].median())}
        for col in use_cols:
            stat[col] = float(g[col].mean())
        cluster_stats.append(stat)

    for col in use_cols:
        vals = [s[col] for s in cluster_stats]
        mx = max(vals) if max(vals) > 0 else 1
        for s in cluster_stats:
            s[col + "_norm"] = s[col] / mx

    return {
        "elbow": elbow, "pca_points": pca_points, "pca_explained": explained,
        "cluster_stats": cluster_stats,
        "feature_names": use_cols, "feature_cn": [COLUMN_CN.get(c, c) for c in use_cols],
    }


def compute_regression(df):
    use_cols = [c for c in BEHAVIOR_FEATURES if c in df.columns]
    X, y = df[use_cols].values, df["grade"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    coefficients = sorted(
        [{"feature": COLUMN_CN.get(c, c), "feature_en": c, "coef": float(reg.coef_[i])}
         for i, c in enumerate(use_cols)],
        key=lambda x: abs(x["coef"]), reverse=True,
    )

    idx = np.arange(len(y_test))
    if len(idx) > 500:
        idx = np.random.RandomState(42).choice(idx, 500, replace=False)

    return {
        "coefficients": coefficients, "intercept": float(reg.intercept_),
        "r2": r2_score(y_test, y_pred), "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "predictions": [{"actual": float(y_test[i]), "predicted": float(y_pred[i])} for i in idx],
        "residuals": (y_test - y_pred)[idx].tolist(),
    }


def compute_classification(df):
    use_cols = [c for c in BEHAVIOR_FEATURES if c in df.columns]
    X, y_raw = df[use_cols].values, df["grade"].values

    threshold = float(np.median(y_raw))
    y = (y_raw >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        threshold = float(np.quantile(y_raw, 0.75))
        y = (y_raw >= threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc, X_test_sc = scaler.fit_transform(X_train), scaler.transform(X_test)

    models_cfg = {
        "logistic": ("逻辑回归", LogisticRegression(max_iter=2000, random_state=42)),
        "decision_tree": ("决策树", DecisionTreeClassifier(max_depth=6, random_state=42)),
        "mlp": ("MLP神经网络", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)),
    }

    results = {"threshold": threshold, "models": {}}
    for name, (cn, model) in models_cfg.items():
        Xtr = X_train_sc if name == "mlp" else X_train
        Xte = X_test_sc if name == "mlp" else X_test
        model.fit(Xtr, y_train)
        yp = model.predict(Xte)

        entry = {
            "cn": cn,
            "accuracy": accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall": recall_score(y_test, yp, zero_division=0),
            "f1": f1_score(y_test, yp, zero_division=0),
            "confusion": confusion_matrix(y_test, yp).tolist(),
        }
        if name == "decision_tree":
            fi = model.feature_importances_
            entry["feature_importance"] = sorted(
                [{"feature": COLUMN_CN.get(c, c), "feature_en": c, "importance": float(fi[i])}
                 for i, c in enumerate(use_cols)],
                key=lambda x: x["importance"], reverse=True,
            )
        results["models"][name] = entry
    return results


# ---------------------------------------------------------------------------
# 构建系统提示词
# ---------------------------------------------------------------------------

def build_system_prompt(active, overview, correlation, group):
    sample_cols = [c for c in BEHAVIOR_FEATURES + ["grade", "region"] if c in active.columns]
    sample_df = active[sample_cols].sample(min(10, len(active)), random_state=42)
    sample_str = sample_df.to_string(index=False, max_colwidth=12)

    corr_str = "\n".join(
        f"  - {g['feature']}（{g['feature_en']}）: r = {g['value']:.4f}"
        for g in correlation["grade_corr"]
    )

    stat_rows = []
    for fs in overview["feat_stats"]:
        stat_rows.append(
            f"| {fs['cn']} | {fs['mean']:.1f} | {fs['median']:.1f} | "
            f"{fs['std']:.1f} | {fs['min']:.0f} | {fs['max']:.0f} |"
        )
    stat_table = "| 特征 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |\n|---|---|---|---|---|---|\n" + "\n".join(stat_rows)

    region_rows = []
    for r in group["region_stats"]:
        region_rows.append(
            f"| {r['region']} | {r['count']} | {r['avg_grade']:.1f} | "
            f"{r['pass_rate']:.1%} | {r['excellent_rate']:.1%} | {r['top_rate']:.1%} |"
        )
    region_table = "| 地区 | 人数 | 平均成绩 | 及格率 | 优秀率 | 拔尖率 |\n|---|---|---|---|---|---|\n" + "\n".join(region_rows)

    span_rows = []
    for s in group["span_stats"]:
        span_rows.append(
            f"| {s['bin']} | {s['count']} | {s['avg_grade']:.1f} | "
            f"{s['pass_rate']:.1%} | {s['excellent_rate']:.1%} | {s['top_rate']:.1%} |"
        )
    span_table = "| 活跃天数 | 人数 | 平均成绩 | 及格率 | 优秀率 | 拔尖率 |\n|---|---|---|---|---|---|\n" + "\n".join(span_rows)

    del_rows = []
    for d in group["deleted_stats"]:
        del_rows.append(
            f"| {d['status']} | {d['count']} | {d['avg_grade']:.1f} | "
            f"{d['pass_rate']:.1%} | {d['excellent_rate']:.1%} | {d['top_rate']:.1%} |"
        )
    del_table = "| 账户状态 | 人数 | 平均成绩 | 及格率 | 优秀率 | 拔尖率 |\n|---|---|---|---|---|---|\n" + "\n".join(del_rows)

    return f"""你是「在线学习行为数据分析平台」的 AI 助手。你正在分析一份 Coursera MOOC（大规模在线开放课程）的学生行为数据集。

## 数据集概况
- 总注册人数: {overview['total_registered']}
- 实际参与学习（成绩>0）: {overview['n_active']} 人（{overview['n_active']/overview['total_registered']:.1%}）
- 退课率: {overview['dropout_rate']:.1%}
- 以下所有数据均基于 {overview['n_active']} 名活跃学生（已剔除 grade=0 的退课者）

## 成绩统计
- 均值: {overview['avg_grade']:.2f}，中位数: {overview['median_grade']:.2f}，标准差: {overview['std_grade']:.2f}
- 及格率（≥60）: {overview['pass_rate']:.1%}
- 优秀率（≥80）: {overview['excellent_rate']:.1%}
- 拔尖率（≥90）: {overview['top_rate']:.1%}

## 数据说明
- `quiz_score_sum` 和 `assignment_score_sum` 因与 grade 循环相关（r≈0.99）已排除
- 仅使用真正的行为特征进行分析

## 行为特征描述性统计
{stat_table}

## 行为特征与成绩的 Pearson 相关系数
{corr_str}

## 地区对比
{region_table}

## 活跃天数与成绩
{span_table}

## 账户状态对比
{del_table}

## 样本数据（随机 10 行）
```
{sample_str}
```

## 核心字段说明
| 列名 | 中文 | 说明 |
|------|------|------|
| duration | 在线时长 | 平台停留总秒数 |
| active_span | 活跃天数 | 最后访问 - 注册日期（天） |
| page_views | 页面浏览 | 浏览页面总次数 |
| lecture_item_views | 课件浏览 | 浏览课件/讲义总次数 |
| video_views | 视频观看 | 观看视频总次数 |
| forum_posts_count | 发帖数 | 论坛发帖数 |
| forum_comments_count | 评论数 | 论坛评论数 |
| grade | 成绩 | 课程最终成绩 |
| region | 地区 | 从 timezone 提取的大洲 |

## 你的职责
- 基于上述数据回答用户关于学习行为与成绩关系的问题
- 用通俗但专业的语言解释统计发现
- 提出有教育意义的洞察和改进建议
- 如果需要更详细的数据才能回答，请诚实说明
- 用中文回答，可使用 Markdown 格式"""


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

CURRENT = {"csv_name": None, "data_json": None, "system_prompt": None}


def run_analysis(df, csv_name):
    print(f"正在分析 {csv_name} ...")

    raw, active, total_n, zero_n = prepare_dataframe(df)
    if len(active) == 0:
        raise ValueError("数据中没有可用的活跃学生（grade 均为 0 或缺失）")

    overview = safe(compute_overview(raw, active, total_n, zero_n))
    correlation = safe(compute_correlation(active))
    scatter = safe(compute_scatter(active))
    group = safe(compute_group_analysis(raw, active))
    clustering = safe(compute_clustering(active))
    regression = safe(compute_regression(active))
    classification = safe(compute_classification(active))
    system_prompt = build_system_prompt(active, overview, correlation, group)

    all_data = {
        "csv_name": csv_name,
        "column_cn": COLUMN_CN,
        "behavior_features": BEHAVIOR_FEATURES,
        "overview": overview,
        "correlation": correlation,
        "scatter": scatter,
        "group": group,
        "clustering": clustering,
        "regression": regression,
        "classification": classification,
    }

    CURRENT["csv_name"] = csv_name
    CURRENT["data_json"] = json.dumps(all_data, ensure_ascii=False)
    CURRENT["system_prompt"] = system_prompt
    print("分析完成。")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", data_json=CURRENT["data_json"] or "null")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"error": "请通过表单字段 'file' 上传文件"}, 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return {"error": "请选择 CSV 文件"}, 400
    try:
        raw_bytes = f.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return {"error": f"CSV 解析失败: {e}"}, 400

    if "grade" not in df.columns:
        return {"error": "CSV 缺少必要的 'grade' 列"}, 400

    try:
        run_analysis(df, f.filename)
    except Exception as e:
        return {"error": f"分析失败: {e}"}, 500
    return {"ok": True, "csv_name": f.filename}


@app.route("/api/reset", methods=["POST"])
def reset():
    CURRENT["csv_name"] = None
    CURRENT["data_json"] = None
    CURRENT["system_prompt"] = None
    return {"ok": True}


@app.route("/api/chat", methods=["POST"])
def chat_proxy():
    body = request.get_json(force=True)
    api_base = body.get("api_base", "").rstrip("/")
    api_key = body.get("api_key", "")
    model = body.get("model", "gpt-4o-mini")
    messages = body.get("messages", [])

    if not api_base or not api_key:
        return {"error": "请先配置 API 地址和密钥"}, 400
    if not CURRENT["system_prompt"]:
        return {"error": "请先上传数据文件"}, 400

    full_messages = [{"role": "system", "content": CURRENT["system_prompt"] or ""}] + messages
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": full_messages, "stream": True, "temperature": 0.7}

    try:
        upstream = http_requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        upstream.raise_for_status()
    except http_requests.RequestException as e:
        return {"error": f"API 请求失败: {e}"}, 502

    def generate():
        for line in upstream.iter_lines():
            if line:
                yield line.decode("utf-8") + "\n"
        yield "data: [DONE]\n"

    return Response(generate(), content_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
