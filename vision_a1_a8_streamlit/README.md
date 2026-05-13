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

