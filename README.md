<div align="center">

# EduAnalytics

**在线学习行为数据分析与 AI 问答平台**

一款拖拽即用的 Flask Web 应用，为 MOOC 学生会话数据提供描述统计、相关性、聚类、回归、分类分析，
并内置基于当前数据集上下文的大语言模型对话助手。

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![ECharts](https://img.shields.io/badge/ECharts-5.x-AA344D)](https://echarts.apache.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://eduanalytics-qpq7.onrender.com/)

🌐 **在线演示**　<https://eduanalytics-qpq7.onrender.com/>　·　无需本地安装，浏览器打开即用

</div>

---

## ✨ 特性

- **拖拽上传**　任意符合规范的 MOOC 会话 CSV 拖入网页即可触发完整分析管线
- **七个分析模块**　概览 / 相关性 / 散点 / 分组对比 / 聚类 / 回归 / 分类
- **数据感知 AI 助手**　集成 OpenAI 兼容 API，系统提示词自动注入当前数据集的描述统计与分组表
- **配置持久化**　API 地址、密钥、模型名写入浏览器 `localStorage`，刷新后自动恢复
- **Claude 风格界面**　暖米色背景、珊瑚色强调、Fraunces 衬线标题，长时间阅读视觉压力低
- **响应式可交互图表**　基于 Apache ECharts 5，支持悬停、缩放、图例筛选、模型切换

## 🗂️ 分析管线

| 模块 | 方法 | 主要产出 |
|------|------|----------|
| 数据概览 | 描述性统计 + 三率（及格/优秀/拔尖） | 六张指标卡、成绩直方图、归一化箱线图 |
| 相关性分析 | Pearson 相关系数 | 七阶热力图、与成绩相关系数排序图 |
| 散点图 | OLS 拟合 | 可切换变量 X 轴、实时相关系数显示 |
| 分组对比 | 地区 / 账户状态 / 活跃天数分箱 | 条形-折线双轴图、退课率图 |
| 聚类分析 | K-Means + PCA + 轮廓系数 | 肘部图、二维聚类散点、特征雷达图 |
| 回归分析 | 线性回归 | 系数条形图、实际 vs 预测对角线图、残差直方图 |
| 分类分析 | 逻辑回归 / 决策树 / MLP | 指标对比柱状图、混淆矩阵、特征重要性 |

## 🚀 快速开始

> 💡 **不想本地安装？** 直接访问线上部署版：<https://eduanalytics-qpq7.onrender.com/>
> （Render 免费实例首次访问可能需 30 秒左右冷启动，请耐心等待）

### 方式 A：使用 [uv](https://docs.astral.sh/uv/)（推荐）

[`uv`](https://docs.astral.sh/uv/) 是 Astral 出品的极速 Python 包管理器，自带虚拟环境管理、依赖解析与 lockfile，一条命令即可跨平台还原完全一致的环境。

```bash
# 克隆仓库
git clone https://github.com/Lee-Eee-Eee/EduAnalytics.git
cd EduAnalytics

# 一键创建虚拟环境并安装锁定的依赖（读取 pyproject.toml + uv.lock）
uv sync

# 启动服务
uv run python app.py
```

没装 `uv` 的先装一下：

- Linux / macOS：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows（PowerShell）：`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- 或通过 pip：`pip install uv`

### 方式 B：传统 venv + pip

```bash
git clone https://github.com/Lee-Eee-Eee/EduAnalytics.git
cd EduAnalytics

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate            # Windows (cmd)
# .venv\Scripts\Activate.ps1        # Windows (PowerShell)

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```

### 访问

服务默认监听 `0.0.0.0:5000`，浏览器打开 <http://localhost:5000> 即可看到上传界面。

如需修改监听端口，可通过 `PORT` 环境变量启动（与云端部署一致）：

```bash
# 例：监听 8080 端口
PORT=8080 python app.py
```

### 使用步骤

1. **上传数据**　把符合格式的 CSV 文件拖入页面中央的虚线框，或点击"选择文件"按钮（最大 64 MB）
2. **浏览分析**　上传成功后页面自动刷新，通过顶部导航栏或滚动浏览七个分析章节
3. **配置 AI 助手**（可选）　点击右上角"AI 助手"，填写 OpenAI 兼容接口的 `base URL`、`API Key`、模型名（首次填写后自动保存至浏览器本地）
4. **开始对话**　输入自然语言问题，助手基于上传的数据集进行回答
5. **更换数据集**　在顶部"当前数据"栏点击"更换文件"清空当前状态并重新上传

## 📦 数据格式

仅 `grade` 列是**必需**的，其余字段越齐全分析越丰富。平台会自动识别并使用以下列：

| 列名 | 含义 | 是否必需 |
|------|------|---------|
| `grade` | 课程最终成绩（0–100） | ✅ 必需 |
| `duration` | 平台停留总秒数 | 推荐 |
| `page_views` | 浏览页面总次数 | 推荐 |
| `lecture_item_views` | 浏览课件 / 讲义总次数 | 推荐 |
| `video_views` | 观看视频总次数 | 推荐 |
| `forum_posts_count` | 论坛发帖数 | 推荐 |
| `forum_comments_count` | 论坛评论数 | 推荐 |
| `registration_time` / `last_access_time` | 时间戳，用于派生 `active_span`（活跃天数） | 可选 |
| `timezone` | 时区，用于提取 `region`（地区） | 可选 |
| `deleted` | 账户状态（0/1） | 可选 |

`grade = 0` 的记录被视为退课 / 未参与，全部分析将剔除后进行。

## 🤖 AI 助手

AI 助手通过 `/api/chat` 将请求代理到任意 **OpenAI Chat Completions 兼容** 的服务（OpenAI、Azure OpenAI、DeepSeek、Moonshot、本地 vLLM / Ollama 的 `/v1` 接口等）。支持 SSE 流式响应。

每次请求时，后端会自动把当前数据集的**概况、描述统计、相关系数、分组统计与样本行**作为 `system` 消息注入，使助手的回答与当前数据强耦合。

配置三项（填写后自动保存到浏览器 `localStorage`）：

- **API 地址**　例如 `https://api.openai.com/v1`
- **API 密钥**　例如 `sk-...`
- **模型**　例如 `gpt-4o-mini`、`deepseek-chat`、`qwen-turbo`

## 🏗️ 项目结构

```
EduAnalytics/
├── app.py              # Flask 后端：路由、数据管线、LLM 代理
├── templates/
│   └── index.html      # 单页前端：拖拽上传 · ECharts · 对话面板
├── pyproject.toml      # 项目元数据与依赖（uv / pip 均可读）
├── uv.lock             # uv 锁定的依赖版本，保证可重现环境
├── requirements.txt    # pip 用户的等价依赖清单
├── README.md
└── .gitignore
```

后端实现简洁：所有分析函数（`compute_overview`, `compute_correlation`, `compute_scatter`, `compute_group_analysis`, `compute_clustering`, `compute_regression`, `compute_classification`）均为纯函数，接收 DataFrame 返回 JSON 可序列化的结果字典，便于单独测试或复用。

## 🎨 设计注记

界面参考了 Anthropic Claude 官方网页的视觉语言：

- **背景**　`#FAF9F5` 暖米色
- **强调色**　`#D97757` 珊瑚橙
- **正文**　`#3D3929` 暖近黑
- **字体**　Fraunces（衬线标题）/ Inter（正文 UI）/ JetBrains Mono（数值徽章）

ECharts 配色与上述色系保持协调，并为不同系列提供鼠绿、薰衣草紫、琥珀金、玫瑰色等暖色补色，避免饱和度过高导致的视觉疲劳。

## 📜 鸣谢

本项目为\
**北京大学 2026 年春季《教育与人工智能》课程**\
作业成果，感谢课程主讲教师**贾积有教授**在研究选题、分析方法与平台设计上的悉心指导。

作者 · [**李涛**](https://github.com/Lee-Eee-Eee) · 清华大学工程物理系

---

<div align="center">

若本项目对你的研究或课程有帮助，欢迎 ⭐ Star 支持。
问题与建议请通过 [Issues](https://github.com/Lee-Eee-Eee/EduAnalytics/issues) 反馈。

</div>
