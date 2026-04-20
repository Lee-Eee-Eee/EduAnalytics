import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression

# 读取数据
data = pd.read_csv('20240110891773837079.csv')


def get_numeric_clean_df(df: pd.DataFrame) -> pd.DataFrame:
	"""保留数值列并处理缺失值，返回可直接建模的数据集。"""
	numeric_df = df.select_dtypes(include=[np.number]).copy()
	# 删除全为空值的列，避免后续填充后仍为 NaN
	numeric_df = numeric_df.dropna(axis=1, how='all')
	# 丢弃标签缺失的样本
	numeric_df = numeric_df.dropna(subset=['grade'])
	# 用中位数填充其余数值缺失值
	numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
	return numeric_df


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
	"""计算成绩与行为变量的相关系数。"""
	corr = df.corr(numeric_only=True)
	corr.to_csv('correlation_matrix.csv', encoding='utf-8-sig')

	grade_corr = corr['grade'].drop('grade').sort_values(ascending=False)
	grade_corr.to_csv('grade_behavior_correlations.csv', encoding='utf-8-sig')

	print('\n[1] 成绩与行为变量相关系数（Top 10 正相关）:')
	print(grade_corr.head(10))
	print('\n[1] 成绩与行为变量相关系数（Top 10 负相关）:')
	print(grade_corr.tail(10))
	return grade_corr


def draw_scatter(df: pd.DataFrame, x_col: str = 'duration') -> None:
	"""画散点图：横轴行为变量，纵轴成绩。"""
	if x_col not in df.columns:
		raise ValueError(f'列 {x_col} 不存在，无法绘图。')

	plt.figure(figsize=(8, 5))
	plt.scatter(df[x_col], df['grade'], alpha=0.35, s=14)
	plt.xlabel(x_col)
	plt.ylabel('grade')
	plt.title(f'grade vs {x_col}')
	plt.grid(alpha=0.2)
	plt.tight_layout()
	plt.savefig('scatter_grade_duration.png', dpi=180)
	plt.close()


def clustering_analysis(df: pd.DataFrame, features: list, n_clusters: int = 3) -> None:
	"""KMeans 聚类，并保存聚类标签和可视化。"""
	use_cols = [c for c in features if c in df.columns]
	if len(use_cols) < 2:
		raise ValueError('可用于聚类的特征不足，至少需要2个数值特征。')

	X = df[use_cols]
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	labels = kmeans.fit_predict(X)

	cluster_df = df.copy()
	cluster_df['cluster'] = labels
	cluster_df[['grade', 'duration', 'cluster']].to_csv(
		'clustering_kmeans.csv', index=False, encoding='utf-8-sig'
	)

	plt.figure(figsize=(8, 5))
	x_axis = 'duration' if 'duration' in cluster_df.columns else use_cols[0]
	plt.scatter(cluster_df[x_axis], cluster_df['grade'], c=cluster_df['cluster'], s=14, alpha=0.5)
	plt.xlabel(x_axis)
	plt.ylabel('grade')
	plt.title('KMeans Clustering Result')
	plt.grid(alpha=0.2)
	plt.tight_layout()
	plt.savefig('clustering_scatter.png', dpi=180)
	plt.close()

	print('\n[2] 聚类完成，各簇样本数量:')
	print(cluster_df['cluster'].value_counts().sort_index())


def regression_and_classification(df: pd.DataFrame, features: list) -> None:
	"""回归与分类：分析成绩受哪些因素影响。"""
	use_cols = [c for c in features if c in df.columns]
	X = df[use_cols]
	y = df['grade']

	# 回归
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	reg = LinearRegression()
	reg.fit(X_train, y_train)
	y_pred = reg.predict(X_test)

	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	coef_series = pd.Series(reg.coef_, index=use_cols).sort_values(ascending=False)

	print('\n[3] 回归模型结果:')
	print(f'MSE = {mse:.4f}')
	print(f'R^2 = {r2:.4f}')
	print('回归系数（从大到小）:')
	print(coef_series)

	with open('regression_metrics.txt', 'w', encoding='utf-8') as f:
		f.write(f'MSE: {mse:.6f}\n')
		f.write(f'R2: {r2:.6f}\n\n')
		f.write('Coefficients:\n')
		f.write(coef_series.to_string())

	# 分类（将成绩二分类，自动确保存在两个类别）
	threshold = float(df['grade'].median())
	y_cls = (df['grade'] > threshold).astype(int)
	if y_cls.nunique() < 2:
		threshold = float(df['grade'].quantile(0.75))
		y_cls = (df['grade'] > threshold).astype(int)
	if y_cls.nunique() < 2:
		threshold = 0.0
		y_cls = (df['grade'] > threshold).astype(int)

	Xc_train, Xc_test, yc_train, yc_test = train_test_split(
		X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
	)
	clf = LogisticRegression(max_iter=2000)
	clf.fit(Xc_train, yc_train)
	yc_pred = clf.predict(Xc_test)

	acc = accuracy_score(yc_test, yc_pred)
	report = str(classification_report(yc_test, yc_pred, digits=4))

	print('\n[3] 分类模型结果（高分/低分）:')
	print(f'阈值(自动选择) = {threshold:.4f}')
	print(f'Accuracy = {acc:.4f}')
	print(report)

	with open('classification_metrics.txt', 'w', encoding='utf-8') as f:
		f.write(f'Threshold(auto): {threshold:.6f}\n')
		f.write(f'Accuracy: {acc:.6f}\n\n')
		f.write(report)


def main() -> None:
	numeric_df = get_numeric_clean_df(data)

	# 可根据需要调整这组行为特征
	behavior_features = [
		'duration',
		'video_views',
		'page_views',
		'assignment_score_sum',
		'lecture_item_views',
		'quiz_score_sum',
		'forum_posts_count',
		'forum_comments_count',
	]

	print('样本数:', len(numeric_df))
	print('可用数值列:', list(numeric_df.columns))

	correlation_analysis(numeric_df)
	draw_scatter(numeric_df, x_col='duration')
	clustering_analysis(numeric_df, features=behavior_features, n_clusters=3)
	regression_and_classification(numeric_df, features=behavior_features)


if __name__ == '__main__':
	main()

