# 模型识别作业 A1-A8：Streamlit 一体化实验平台

这个仓库把 PDF 中的八个 Vibe Coding 作业合并成一个可交互的 Streamlit Web App，入口文件为 `streamlit_app.py`。

- A1：RGB/HSV 颜色空间通道分解，最近邻/双线性/双三次/Lanczos 插值，放大、缩小、旋转。
- A2：Box/Gaussian/Median/Sobel/Laplacian 空间滤波，ROI 梯度方向，FFT 频域低通/高通/带通滤波，频谱比较。
- A3：Canny，Harris/SIFT/ORB 特征点，双图匹配，RANSAC，全景拼接。
- A4：最小二乘线性回归，KNN 与线性分类器，模板图像可视化，SGD/loss 曲线。
- A5：HOG + Bag of Words + SVM，反向传播演示，Tiny CNN，Plain/Residual 网络深度比较。
- A6：Tiny FCN 语义分割，R-CNN/Fast/Faster R-CNN 教学版检测流程，Mask R-CNN 式实例分割，性能对比表。
- A7：旋转预测、拼图、遮挡/MAE、伪彩色化、SimCLR 简化对比学习，loss/accuracy 可视化。
- A8：Autoencoder 与 VAE 重构对比，误差热力图，二维潜空间散点，潜空间插值，轻量 GAN，prompt/negative/guidance 参数实验。

> 使用的 Agent/LLM：GPT-5.5 Pro + Python/OpenCV/scikit-learn/PyTorch/Streamlit。

## 1. 本地运行

建议使用 Python 3.11 或 3.12。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

浏览器打开本地地址后，在左侧导航选择 A1-A8 页面。A5-A8 含深度学习训练，云端或低配置电脑建议把 epoch 调小。

## 2. 项目结构

```text
.
├── streamlit_app.py          # Streamlit 主程序，八个作业页面都在这里
├── requirements.txt          # Python 依赖，Streamlit Cloud 会自动安装
├── .streamlit/config.toml    # 上传大小等配置
├── reports/                  # 八个 PDF 报告和 Markdown 报告模板
├── sample_outputs/           # 生成报告用的示例结果图
└── README.md
```

## 3. 如何提交作业

每个作业建议提交三类内容：

1. PDF 实验报告：`reports/A1_report.pdf` 到 `reports/A8_report.pdf`。
2. 核心源码：`streamlit_app.py`，或者把整份仓库打包上传。
3. 可外链 URL：部署成功后把 Streamlit Cloud 的公开链接提交。

如果老师要求每个作业单独提交源码，可以直接在 `streamlit_app.py` 中搜索 `page_a1`、`page_a2`……`page_a8`，对应函数就是每个作业的核心实现。

## 4. 部署到 GitHub + Streamlit Community Cloud

### 4.1 新建 GitHub 仓库并上传

在 GitHub 网页端新建仓库，例如 `model-recognition-a1-a8`。在本地项目目录执行：

```bash
git init
git add .
git commit -m "finish A1-A8 model recognition streamlit app"
git branch -M main
git remote add origin https://github.com/你的用户名/model-recognition-a1-a8.git
git push -u origin main
```

### 4.2 Streamlit Cloud 部署

1. 登录 Streamlit Community Cloud。
2. 连接 GitHub 账号。
3. 点击 Create app。
4. Repository 选择刚才的仓库。
5. Branch 填 `main`。
6. Main file path 填 `streamlit_app.py`。
7. Advanced settings 里选择 Python 3.12 或 3.11。
8. 点击 Deploy。

部署完成后得到形如 `https://xxxx.streamlit.app/` 的 URL，把这个 URL 放到作业提交处。

## 5. 常见问题

### 5.1 PyTorch 安装慢或云端资源不足

A5-A8 用到了 PyTorch。若部署时安装过慢或云端内存不足，可先把 A5-A8 页面的 epoch 调到较小，或者把 `requirements.txt` 中的 `torch` 改成 CPU-only 官方安装方式。课堂演示通常只需运行低 epoch 版本即可。

### 5.2 OpenCV 在云端报 libGL 错误

本项目使用的是 `opencv-python-headless`，通常不会出现桌面 GUI 依赖问题。不要把它替换成 `opencv-python`。

### 5.3 图片上传太大

`.streamlit/config.toml` 已把上传上限设置为 100MB。建议实际实验中上传 1-5MB 的图片，避免云端处理变慢。

## 6. 报告截图建议

每个页面都至少截 3 张图：

- 页面输入区域和参数设置。
- 核心算法输出图。
- 指标表、loss 曲线或对比结果。

报告中已经放了示例输出图；正式提交前，可以用你自己部署后的页面截图替换这些图。
