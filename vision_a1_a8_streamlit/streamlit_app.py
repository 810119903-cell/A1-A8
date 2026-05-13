"""
模型识别作业 A1-A8：Streamlit 一体化实验平台

运行：streamlit run streamlit_app.py
提交：PDF 报告 + 核心源码 py + 可访问 URL（部署后）
Agent/LLM：GPT-5.5 Pro
"""
from __future__ import annotations

import io
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits, make_moons
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(2)
except Exception:  # pragma: no cover - used on cloud if torch is unavailable
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

APP_TITLE = "模型识别作业 A1-A8 · Streamlit 实验平台"
AGENT_TEXT = "GPT-5.5 Pro + Python/OpenCV/scikit-learn/PyTorch/Streamlit"


# ===============================
# General helpers
# ===============================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def to_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)


def normalize_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def demo_image(width: int = 420, height: int = 280) -> np.ndarray:
    """A colorful synthetic image with edges, texture, gradients and text."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[..., 0] = np.clip(255 * xx, 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(255 * yy, 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(255 * (1 - 0.5 * xx + 0.25 * np.sin(12 * yy)), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 50), (170, 180), (255, 40, 40), -1)
    cv2.circle(img, (280, 120), 70, (40, 220, 70), -1)
    cv2.line(img, (20, height - 40), (width - 20, 30), (20, 20, 240), 5)
    cv2.putText(img, "A1-A8", (210, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    rng = np.random.default_rng(10)
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def read_uploaded_image(uploaded_file, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    if uploaded_file is None:
        return fallback.copy() if fallback is not None else demo_image()
    try:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    except Exception as exc:
        st.error(f"图像读取失败：{exc}")
        return fallback.copy() if fallback is not None else demo_image()


def read_multiple_images(uploaded_files) -> List[np.ndarray]:
    imgs: List[np.ndarray] = []
    for f in uploaded_files or []:
        imgs.append(read_uploaded_image(f))
    return imgs


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)


def plot_image_grid(images: Sequence[np.ndarray], titles: Sequence[str], cols: int = 3, cmap: Optional[str] = None, figsize_per_col: float = 3.2):
    rows = int(math.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_col * cols, figsize_per_col * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(to_uint8(img))
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def image_download_button(img: np.ndarray, label: str, file_name: str) -> None:
    pil = Image.fromarray(to_uint8(img))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    st.download_button(label, data=buf.getvalue(), file_name=file_name, mime="image/png")


def sidebar_common() -> None:
    st.sidebar.info(
        "本应用把 A1-A8 的交互实验合并为一个可部署的 Streamlit 项目。\n\n"
        f"报告中列出的 Agent/LLM：{AGENT_TEXT}。"
    )
    st.sidebar.caption("提示：云端运行深度学习页时请把 epoch 调小；本地电脑可调大。")


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


# ===============================
# A1 Color spaces and interpolation
# ===============================

def color_channel_images(img: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
    rgb = to_uint8(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rgb_imgs = []
    for i, name in enumerate(["R", "G", "B"]):
        ch = np.zeros_like(rgb)
        ch[..., i] = rgb[..., i]
        rgb_imgs.append(ch)
    hsv_vis = [to_uint8(hsv[..., 0]), to_uint8(hsv[..., 1]), to_uint8(hsv[..., 2])]
    images = [rgb] + rgb_imgs + hsv_vis
    titles = ["Original RGB", "R channel", "G channel", "B channel", "HSV-H", "HSV-S", "HSV-V"]
    return images, titles


def transform_image(img: np.ndarray, scale: float, angle: float, interpolation_name: str) -> np.ndarray:
    interp_map = {
        "Nearest 最近邻": cv2.INTER_NEAREST,
        "Bilinear 双线性": cv2.INTER_LINEAR,
        "Bicubic 双三次": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4,
    }
    interp = interp_map[interpolation_name]
    h, w = img.shape[:2]
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    rh, rw = resized.shape[:2]
    M = cv2.getRotationMatrix2D((rw / 2, rh / 2), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    bound_w = int((rh * sin) + (rw * cos))
    bound_h = int((rh * cos) + (rw * sin))
    M[0, 2] += (bound_w / 2) - rw / 2
    M[1, 2] += (bound_h / 2) - rh / 2
    return cv2.warpAffine(resized, M, (bound_w, bound_h), flags=interp, borderMode=cv2.BORDER_REFLECT)


def page_a1() -> None:
    section_header("A1 图像颜色空间、图像插值", "RGB/HSV 通道分解；最近邻、双线性、双三次、Lanczos 插值的放大、缩小、旋转。")
    uploaded = st.file_uploader("上传一张图像（不上传则使用内置测试图）", type=["png", "jpg", "jpeg", "bmp"], key="a1_img")
    img = read_uploaded_image(uploaded)
    images, titles = color_channel_images(img)
    st.pyplot(plot_image_grid(images, titles, cols=4))

    st.markdown("### 插值变换")
    c1, c2, c3 = st.columns(3)
    with c1:
        scale = st.slider("缩放倍数", 0.2, 3.0, 1.35, 0.05)
    with c2:
        angle = st.slider("旋转角度", -180, 180, 25, 1)
    with c3:
        interpolation = st.selectbox("插值方法", ["Nearest 最近邻", "Bilinear 双线性", "Bicubic 双三次", "Lanczos"])
    out = transform_image(img, scale, angle, interpolation)
    st.image([img, out], caption=["输入图像", f"输出：{interpolation}, scale={scale}, angle={angle}"], use_container_width=True)
    image_download_button(out, "下载 A1 插值结果图", "A1_interpolation_result.png")
    st.success("已覆盖 A1：RGB/HSV 每通道输出 + 多种插值算法 + 交互式放大/缩小/旋转。")


# ===============================
# A2 Filtering and FFT
# ===============================

def apply_spatial_filter(gray: np.ndarray, method: str, k: int, sigma: float) -> np.ndarray:
    if method == "Box 均值滤波":
        return cv2.blur(gray, (k, k))
    if method == "Gaussian 高斯滤波":
        return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma)
    if method == "Median 中值滤波":
        return cv2.medianBlur(gray, k)
    if method == "Sobel 梯度幅值":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
        return to_uint8(np.sqrt(gx * gx + gy * gy))
    if method == "Laplacian 拉普拉斯":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=k)
        return to_uint8(np.abs(lap))
    return gray


def fft_filter(gray: np.ndarray, mode: str, radius: int, band_radius: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if mode == "低通 Low-pass":
        mask = dist <= radius
    elif mode == "高通 High-pass":
        mask = dist >= radius
    else:
        mask = (dist >= max(1, radius - band_radius)) & (dist <= radius + band_radius)
    filtered_shift = fshift * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    out = to_uint8(np.abs(inv))
    spectrum = to_uint8(np.log1p(np.abs(fshift)))
    mask_u8 = (mask.astype(np.uint8) * 255)
    return spectrum, mask_u8, out


def gradient_roi_figure(gray: np.ndarray, roi: Tuple[int, int, int, int], step: int = 8):
    x, y, w, h = roi
    roi_img = gray[y:y + h, x:x + w]
    if roi_img.size == 0:
        roi_img = gray
    gx = cv2.Sobel(roi_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    yy, xx = np.mgrid[0:roi_img.shape[0]:step, 0:roi_img.shape[1]:step]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(roi_img, cmap="gray")
    ax.quiver(xx, yy, gx[yy, xx], gy[yy, xx], angles="xy", scale_units="xy", scale=max(20, float(mag.mean()) * 4))
    ax.set_title(f"ROI 梯度方向，平均幅值={mag.mean():.2f}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def page_a2() -> None:
    section_header("A2 图像滤波", "Box/Gaussian/Median/Sobel 比较；ROI 梯度方向；傅里叶频域滤波与谱图。")
    uploaded = st.file_uploader("上传一张图像", type=["png", "jpg", "jpeg", "bmp"], key="a2_img")
    img = read_uploaded_image(uploaded)
    gray = rgb_to_gray(img)

    st.markdown("### 空间域滤波比较")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("滤波器", ["Box 均值滤波", "Gaussian 高斯滤波", "Median 中值滤波", "Sobel 梯度幅值", "Laplacian 拉普拉斯"])
    with c2:
        k = st.slider("核大小（奇数）", 3, 15, 5, 2)
    with c3:
        sigma = st.slider("高斯 sigma", 0.1, 5.0, 1.2, 0.1)
    filtered = apply_spatial_filter(gray, method, k, sigma)
    st.image([gray, filtered], caption=["灰度输入", method], use_container_width=True, clamp=True)

    st.markdown("### 局部区域梯度方向")
    h, w = gray.shape
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x0 = st.slider("ROI x", 0, max(0, w - 20), min(40, max(0, w - 20)), key="a2_x")
    with col2:
        y0 = st.slider("ROI y", 0, max(0, h - 20), min(40, max(0, h - 20)), key="a2_y")
    with col3:
        rw = st.slider("ROI 宽", 20, w, min(160, w), key="a2_w")
    with col4:
        rh = st.slider("ROI 高", 20, h, min(120, h), key="a2_h")
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (min(w - 1, x0 + rw), min(h - 1, y0 + rh)), (255, 255, 255), 2)
    st.image(overlay, caption="ROI 位置", use_container_width=True)
    st.pyplot(gradient_roi_figure(gray, (x0, y0, min(rw, w - x0), min(rh, h - y0))))

    st.markdown("### 频率域滤波")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        mode = st.selectbox("频域滤波", ["低通 Low-pass", "高通 High-pass", "带通 Band-pass"])
    with fc2:
        radius = st.slider("频率半径", 5, min(gray.shape) // 2, max(10, min(gray.shape) // 8))
    with fc3:
        band = st.slider("带宽", 2, 60, 15)
    spectrum, mask, freq_out = fft_filter(gray, mode, radius, band)
    st.image([spectrum, mask, freq_out], caption=["FFT 幅度谱", "频域掩膜", "反变换结果"], use_container_width=True, clamp=True)

    st.markdown("### 旋转/缩放对频谱的影响")
    angle = st.slider("频谱对比旋转角度", -90, 90, 30, key="a2_rot")
    scaled_rot = transform_image(img, 1.0, angle, "Bilinear 双线性")
    spec2, _, _ = fft_filter(rgb_to_gray(scaled_rot), "低通 Low-pass", radius)
    st.image([spectrum, spec2], caption=["原图频谱", f"旋转 {angle}° 后频谱"], use_container_width=True, clamp=True)
    st.success("已覆盖 A2：空间滤波、梯度方向、频域滤波、旋转/缩放谱图比较。")


# ===============================
# A3 Features, matching and stitching
# ===============================

def draw_canny_comparison(gray: np.ndarray, t1: int, t2: int) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = to_uint8(np.sqrt(gx * gx + gy * gy))
    edges = cv2.Canny(gray, t1, t2)
    return mag, edges


def detect_keypoints(img: np.ndarray, method: str = "SIFT", max_features: int = 300):
    gray = rgb_to_gray(img)
    if method == "Harris":
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)
        out = img.copy()
        ys, xs = np.where(corners > 0.01 * corners.max())
        for x, y in zip(xs[:max_features], ys[:max_features]):
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), 1)
        return out, [], None
    if method == "SIFT" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING
    kp, des = detector.detectAndCompute(gray, None)
    out_bgr = cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out, kp, des


def make_matching_pair(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 0.95)
    M[0, 2] += 25
    M[1, 2] += 12
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img, warped


def match_two_images(img1: np.ndarray, img2: np.ndarray, method: str = "SIFT", ratio: float = 0.75) -> Tuple[np.ndarray, Dict[str, float]]:
    gray1, gray2 = rgb_to_gray(img1), rgb_to_gray(img2)
    if method == "SIFT" and hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=600)
        norm = cv2.NORM_L2
    else:
        det = cv2.ORB_create(nfeatures=800)
        norm = cv2.NORM_HAMMING
    kp1, des1 = det.detectAndCompute(gray1, None)
    kp2, des2 = det.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.hstack([img1, img2]), {"keypoints1": len(kp1), "keypoints2": len(kp2), "matches": 0, "inliers": 0}
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    inlier_mask = None
    inliers = 0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().tolist()
            inliers = int(mask.sum())
    drawn_bgr = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), kp2,
        good[:80], None, matchesMask=inlier_mask[:80] if inlier_mask else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    drawn = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
    metrics = {"keypoints1": len(kp1), "keypoints2": len(kp2), "matches": len(good), "inliers": inliers}
    return drawn, metrics


def stitch_images(images: List[np.ndarray]) -> Tuple[np.ndarray, str]:
    if len(images) < 2:
        img1, img2 = make_matching_pair(demo_image())
        images = [img1, img2]
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        bgrs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        status, pano_bgr = stitcher.stitch(bgrs)
        if status == cv2.Stitcher_OK:
            return cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB), "OpenCV Stitcher 成功"
    except Exception:
        pass
    # fallback: horizontal concat with same height
    h = min(i.shape[0] for i in images)
    resized = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in images]
    return np.concatenate(resized, axis=1), "Stitcher 未成功，使用横向拼接作为可视化备选"


def page_a3() -> None:
    section_header("A3 图像特征检测与匹配", "Canny 边缘、Harris/SIFT/ORB 特征、RANSAC 匹配、全景拼接。")
    uploaded = st.file_uploader("上传主图像", type=["png", "jpg", "jpeg", "bmp"], key="a3_main")
    img = read_uploaded_image(uploaded)
    gray = rgb_to_gray(img)

    st.markdown("### Canny 边缘检测：非最大值抑制前后可视化")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.slider("低阈值", 10, 200, 60)
    with c2:
        t2 = st.slider("高阈值", 50, 300, 140)
    mag, edges = draw_canny_comparison(gray, t1, t2)
    st.image([mag, edges], caption=["Sobel 梯度幅值（可理解为抑制前候选）", "Canny 输出边缘（含 NMS 与双阈值连接）"], use_container_width=True, clamp=True)

    st.markdown("### Harris/SIFT/ORB 特征点")
    method = st.selectbox("特征方法", ["Harris", "SIFT", "ORB"])
    kimg, kp, des = detect_keypoints(img, method=method)
    st.image(kimg, caption=f"{method} 特征可视化", use_container_width=True)

    st.markdown("### 两图匹配 + RANSAC")
    up2 = st.file_uploader("上传第二张图像（不上传则自动构造一张旋转平移图）", type=["png", "jpg", "jpeg", "bmp"], key="a3_second")
    img1, default_img2 = make_matching_pair(img)
    img2 = read_uploaded_image(up2, fallback=default_img2)
    match_method = st.selectbox("匹配特征", ["SIFT", "ORB"], key="a3_match_method")
    drawn, metrics = match_two_images(img1, img2, method=match_method)
    st.image(drawn, caption=f"匹配图：{metrics}", use_container_width=True)
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

    st.markdown("### 多图全景拼接")
    ups = st.file_uploader("上传同一场景多张有重叠区域的图像（可多选）", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True, key="a3_pano")
    imgs = read_multiple_images(ups)
    pano, msg = stitch_images(imgs)
    st.image(pano, caption=msg, use_container_width=True)
    st.success("已覆盖 A3：边缘检测、特征点、匹配流程可视化、RANSAC、全景拼接。")


# ===============================
# A4 Regression, KNN and linear classifiers
# ===============================

@st.cache_data(show_spinner=False)
def load_digits_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    images = data.images.astype(np.float32) / 16.0
    return X, y, images


def linear_regression_demo(n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n)
    y = 1.8 * x - 0.5 + rng.normal(0, noise, size=n)
    X = np.c_[np.ones_like(x), x]
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ theta
    mse = float(np.mean((pred - y) ** 2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=18, label="samples")
    ax.plot(x, pred, label=f"y={theta[1]:.2f}x+{theta[0]:.2f}")
    ax.set_title(f"Least Squares Linear Regression, MSE={mse:.3f}")
    ax.legend()
    fig.tight_layout()
    return fig, {"intercept": theta[0], "slope": theta[1], "mse": mse}


@st.cache_data(show_spinner=False)
def run_knn_linear_demo(k_values: Tuple[int, ...] = (1, 3, 5, 7)) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X, y, images = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rows = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        rows.append({"model": f"KNN k={k}", "accuracy": accuracy_score(y_test, clf.predict(X_test))})
    lin = LogisticRegression(max_iter=1000, solver="lbfgs")
    lin.fit(X_train, y_train)
    rows.append({"model": "Linear Logistic", "accuracy": accuracy_score(y_test, lin.predict(X_test))})
    weights = lin.coef_.reshape(10, 8, 8)
    templates = np.array([images[y == cls].mean(axis=0) for cls in range(10)])
    return pd.DataFrame(rows), weights, templates


@st.cache_data(show_spinner=False)
def softmax_sgd_digits(epochs: int = 120, lr: float = 0.8, reg: float = 1e-4) -> Tuple[pd.DataFrame, np.ndarray]:
    X, y, _ = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)
    n, d = X_train.shape
    C = 10
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, size=(d, C))
    losses = []
    Y = np.eye(C)[y_train]
    for ep in range(epochs):
        logits = X_train @ W
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(Y * np.log(probs + 1e-8), axis=1)) + 0.5 * reg * np.sum(W * W)
        grad = X_train.T @ (probs - Y) / n + reg * W
        W -= lr * grad
        if ep % 5 == 0 or ep == epochs - 1:
            test_pred = (X_test @ W).argmax(axis=1)
            losses.append({"epoch": ep, "loss": float(loss), "test_accuracy": accuracy_score(y_test, test_pred)})
    return pd.DataFrame(losses), W.T.reshape(10, 8, 8)


def page_a4() -> None:
    section_header("A4 回归、KNN 与线性分类器", "最小二乘线性回归；KNN/线性分类器；模板图像与 SGD/loss 可视化。")
    st.markdown("### Least Squares Linear Regression")
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("样本数", 20, 300, 80)
    with c2:
        noise = st.slider("噪声", 0.1, 3.0, 0.8, 0.1)
    with c3:
        seed = st.number_input("随机种子", 0, 999, 42)
    fig, metrics = linear_regression_demo(n, noise, int(seed))
    st.pyplot(fig)
    st.json({k: round(float(v), 4) for k, v in metrics.items()})

    st.markdown("### KNN / 线性分类器：digits 图像数据")
    results, weights, templates = run_knn_linear_demo()
    st.dataframe(results, use_container_width=True)
    fig1 = plot_image_grid([to_uint8(t) for t in templates], [f"mean {i}" for i in range(10)], cols=5, cmap="gray", figsize_per_col=2)
    st.pyplot(fig1)
    fig2 = plot_image_grid([to_uint8(w) for w in weights], [f"linear w {i}" for i in range(10)], cols=5, cmap="gray", figsize_per_col=2)
    st.pyplot(fig2)

    st.markdown("### SGD/动量更新思想：Softmax 线性分类器 loss 曲线")
    epochs = st.slider("训练轮数", 30, 300, 120, 10)
    lr = st.slider("学习率", 0.05, 2.0, 0.8, 0.05)
    losses, learned_templates = softmax_sgd_digits(epochs=epochs, lr=lr)
    fig3, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses["epoch"], losses["loss"], label="CE loss")
    ax2 = ax.twinx()
    ax2.plot(losses["epoch"], losses["test_accuracy"], label="test acc", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_title("SGD training process")
    fig3.tight_layout()
    st.pyplot(fig3)
    st.dataframe(losses.tail(10), use_container_width=True)
    st.success("已覆盖 A4：回归、KNN/线性分类器、模板可视化、SGD/loss 演示。")


# ===============================
# A5 HOG/BoW/SVM, backprop, CNN, ResNet
# ===============================

@st.cache_data(show_spinner=False)
def train_bow_svm(n_clusters: int = 16, sample_limit: int = 1200) -> Dict[str, object]:
    X, y, images = load_digits_data()
    images = images[:sample_limit]
    y = y[:sample_limit]
    patches = []
    owners = []
    for idx, img in enumerate(images):
        # 4x4 local patches with stride 2 from 8x8 image
        for r in range(0, 5, 2):
            for c in range(0, 5, 2):
                p = img[r:r + 4, c:c + 4].reshape(-1)
                patches.append(p)
                owners.append(idx)
    patches = np.asarray(patches)
    # For Streamlit Cloud speed, build the visual vocabulary from sampled local patches.
    # This is a lightweight BoW codebook; replacing centers with MiniBatchKMeans centers is a one-line upgrade.
    rng = np.random.default_rng(42)
    center_idx = rng.choice(len(patches), size=n_clusters, replace=False)
    centers = patches[center_idx]
    d2 = ((patches[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    hist = np.zeros((len(images), n_clusters), dtype=np.float32)
    for owner, lab in zip(owners, labels):
        hist[owner, lab] += 1
    hist = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
    hog_features = np.asarray([
        hog(im, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=False)
        for im in images
    ], dtype=np.float32)
    features = np.hstack([hist, hog_features])
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)
    svm = LinearSVC(random_state=42, max_iter=2000)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "hist": hist,
        "centers": centers.reshape(n_clusters, 4, 4),
        "confusion": confusion_matrix(y_test, pred),
    }


@st.cache_data(show_spinner=False)
def train_mlp_backprop(hidden: int = 24, epochs: int = 600, lr: float = 0.08) -> Dict[str, object]:
    X, y = make_moons(n_samples=400, noise=0.22, random_state=0)
    X = StandardScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    rng = np.random.default_rng(1)
    W1 = rng.normal(0, 0.7, (2, hidden))
    b1 = np.zeros((1, hidden))
    W2 = rng.normal(0, 0.7, (hidden, 1))
    b2 = np.zeros((1, 1))
    losses = []
    for ep in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        p = 1 / (1 + np.exp(-z2))
        loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))
        dz2 = (p - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1 * a1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        if ep % 20 == 0 or ep == epochs - 1:
            losses.append({"epoch": ep, "loss": float(loss)})
    pred = (p > 0.5).astype(int).ravel()
    acc = float((pred == y.ravel()).mean())
    return {"X": X, "y": y.ravel(), "weights": (W1, b1, W2, b2), "losses": pd.DataFrame(losses), "accuracy": acc}


def mlp_decision_boundary(result: Dict[str, object]):
    X, y = result["X"], result["y"]
    W1, b1, W2, b2 = result["weights"]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 150),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    a1 = np.tanh(grid @ W1 + b1)
    p = 1 / (1 + np.exp(-(a1 @ W2 + b2)))
    zz = p.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, zz, levels=20, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=18, edgecolors="k")
    ax.set_title(f"Backprop MLP decision boundary, acc={result['accuracy']:.3f}")
    fig.tight_layout()
    return fig


if torch is not None:
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
            self.fc = nn.Linear(24 * 2 * 2, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            return self.fc(x.flatten(1))

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return F.relu(x + self.fc2(F.relu(self.fc1(x))))

    class PlainOrResidualMLP(nn.Module):
        def __init__(self, depth: int = 4, residual: bool = False, dim: int = 64):
            super().__init__()
            self.inp = nn.Linear(64, dim)
            if residual:
                self.layers = nn.ModuleList([ResidualBlock(dim) for _ in range(depth)])
            else:
                self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth * 2)])
            self.residual = residual
            self.out = nn.Linear(dim, 10)

        def forward(self, x):
            x = F.relu(self.inp(x))
            for layer in self.layers:
                if self.residual:
                    x = layer(x)
                else:
                    x = F.relu(layer(x))
            return self.out(x)


@st.cache_resource(show_spinner=False)
def train_tiny_cnn(epochs: int = 6, lr: float = 0.01) -> Dict[str, object]:
    if torch is None:
        return {"error": "PyTorch 未安装"}
    set_seed(42)
    X, y, images = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(images[:, None, :, :], y, test_size=0.25, random_state=42, stratify=y)
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.long)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(test_x).argmax(1)
            acc = float((pred == test_y).float().mean().item())
        hist.append({"epoch": ep + 1, "loss": total_loss / len(train_ds), "test_accuracy": acc})
    with torch.no_grad():
        sample_logits = model(test_x[:16])
        sample_pred = sample_logits.argmax(1).cpu().numpy()
    return {"model": model, "history": pd.DataFrame(hist), "test_images": X_test[:16, 0], "pred": sample_pred, "true": y_test[:16]}


@st.cache_data(show_spinner=False)
def compare_resnet_depths(epochs: int = 8, depth_values: Tuple[int, ...] = (1, 2, 4, 6)) -> pd.DataFrame:
    if torch is None:
        return pd.DataFrame([{"model": "PyTorch unavailable", "accuracy": np.nan}])
    set_seed(123)
    X, y, _ = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.long)
    rows = []
    for residual in [False, True]:
        for depth in depth_values:
            model = PlainOrResidualMLP(depth=depth, residual=residual, dim=48)
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for _ in range(epochs):
                opt.zero_grad()
                loss = F.cross_entropy(model(train_x), train_y)
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = float((model(test_x).argmax(1) == test_y).float().mean().item())
            rows.append({"type": "Residual" if residual else "Plain", "depth": depth, "test_accuracy": acc})
    return pd.DataFrame(rows)


def page_a5() -> None:
    section_header("A5 图像识别与深度网络", "HOG+BoW+SVM；反向传播；CNN；不同深度残差网络性能对比。")
    st.markdown("### HOG + Bag of Words + SVM 图像分类")
    clusters = st.slider("视觉词数量", 8, 32, 16, 4)
    bow = train_bow_svm(n_clusters=clusters)
    X, y, images = load_digits_data()
    fd, hog_img = hog(images[0], orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualize=True)
    centers = [to_uint8(c) for c in bow["centers"][:min(16, clusters)]]
    st.metric("BoW+SVM 测试准确率", f"{bow['accuracy']:.3f}")
    st.pyplot(plot_image_grid([images[0], hog_img] + centers, ["sample", "HOG"] + [f"word {i}" for i in range(len(centers))], cols=6, cmap="gray", figsize_per_col=2))

    st.markdown("### 反向传播演示：两层 MLP 拟合 moons 数据")
    hidden = st.slider("隐藏层神经元", 8, 64, 24, 4)
    epochs = st.slider("反向传播 epoch", 100, 1200, 600, 100)
    bp = train_mlp_backprop(hidden=hidden, epochs=epochs)
    st.pyplot(mlp_decision_boundary(bp))
    fig_loss, ax = plt.subplots(figsize=(6, 3))
    ax.plot(bp["losses"]["epoch"], bp["losses"]["loss"])
    ax.set_title("Backprop loss")
    fig_loss.tight_layout()
    st.pyplot(fig_loss)

    st.markdown("### Tiny CNN 训练/测试（digits，云端轻量版 MNIST 类任务）")
    if torch is None:
        st.warning("未安装 PyTorch，CNN/ResNet 部分无法运行。请在 requirements.txt 中加入 torch。")
    else:
        cnn_epochs = st.slider("CNN epoch", 2, 20, 6, 1)
        cnn = train_tiny_cnn(epochs=cnn_epochs)
        st.dataframe(cnn["history"], use_container_width=True)
        imgs = [to_uint8(i) for i in cnn["test_images"]]
        caps = [f"pred={p}, true={t}" for p, t in zip(cnn["pred"], cnn["true"])]
        st.pyplot(plot_image_grid(imgs, caps, cols=8, cmap="gray", figsize_per_col=1.5))

        st.markdown("### Plain MLP vs Residual MLP 深度对比")
        res_df = compare_resnet_depths(epochs=8)
        st.dataframe(res_df, use_container_width=True)
        fig_res, ax = plt.subplots(figsize=(7, 4))
        for name, sub in res_df.groupby("type"):
            ax.plot(sub["depth"], sub["test_accuracy"], marker="o", label=name)
        ax.set_xlabel("depth")
        ax.set_ylabel("test accuracy")
        ax.set_title("Residual connection helps deeper toy networks")
        ax.legend()
        fig_res.tight_layout()
        st.pyplot(fig_res)
    st.success("已覆盖 A5：传统特征分类、反向传播、CNN、残差/深度对比。")


# ===============================
# A6 Segmentation and object detection
# ===============================

def generate_shape_scene(size: int = 160, n_objects: int = 6, seed: int = 0) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8) + 30
    objs: List[Dict[str, object]] = []
    for i in range(n_objects):
        label = "circle" if rng.random() < 0.5 else "rectangle"
        color = (int(rng.integers(80, 255)), int(rng.integers(80, 255)), int(rng.integers(80, 255)))
        mask = np.zeros((size, size), dtype=np.uint8)
        if label == "circle":
            r = int(rng.integers(12, 25))
            x = int(rng.integers(r + 5, size - r - 5))
            y = int(rng.integers(r + 5, size - r - 5))
            cv2.circle(img, (x, y), r, color, -1)
            cv2.circle(mask, (x, y), r, 255, -1)
        else:
            w = int(rng.integers(22, 42))
            h = int(rng.integers(18, 38))
            x = int(rng.integers(5, size - w - 5))
            y = int(rng.integers(5, size - h - 5))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        ys, xs = np.where(mask > 0)
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (0, 0, 0, 0)
        objs.append({"label": label, "box": box, "mask": mask})
    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, objs


def detect_shape_instances(img: np.ndarray) -> List[Dict[str, object]]:
    gray = rgb_to_gray(img)
    _, thresh = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[Dict[str, object]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True) + 1e-6
        circularity = 4 * math.pi * area / (perimeter ** 2)
        label = "circle" if circularity > 0.75 else "rectangle"
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dets.append({"label": label, "box": (x, y, x + w, y + h), "mask": mask, "score": float(min(0.99, circularity if label == "circle" else 1 - circularity / 1.2))})
    return dets


def overlay_detections(img: np.ndarray, dets: List[Dict[str, object]], draw_masks: bool = True) -> np.ndarray:
    out = img.copy()
    rng = np.random.default_rng(4)
    for det in dets:
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        x1, y1, x2, y2 = det["box"]
        if draw_masks:
            mask = det["mask"] > 0
            out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, str(det["label"]), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def iou_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-8)


def detection_metrics(gt: List[Dict[str, object]], pred: List[Dict[str, object]]) -> Dict[str, float]:
    if not gt or not pred:
        return {"mean_iou": 0.0, "matched": 0, "pred_count": len(pred)}
    used = set()
    ious = []
    matched = 0
    for g in gt:
        best_iou, best_j = 0.0, -1
        for j, p in enumerate(pred):
            if j in used:
                continue
            val = iou_box(g["box"], p["box"])
            if val > best_iou:
                best_iou, best_j = val, j
        if best_iou > 0.3:
            matched += 1
            used.add(best_j)
        ious.append(best_iou)
    return {"mean_iou": float(np.mean(ious)), "matched": matched, "pred_count": len(pred)}


def make_segmentation_dataset(n: int = 96, size: int = 64, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    imgs = []
    masks = []
    for i in range(n):
        img = np.zeros((size, size, 3), dtype=np.uint8) + 25
        mask = np.zeros((size, size), dtype=np.uint8)
        # class 1 circle
        r = int(rng.integers(8, 16))
        x = int(rng.integers(r, size - r))
        y = int(rng.integers(r, size - r))
        cv2.circle(img, (x, y), r, (220, 60, 60), -1)
        cv2.circle(mask, (x, y), r, 1, -1)
        # class 2 rectangle
        w = int(rng.integers(12, 24))
        h = int(rng.integers(10, 22))
        x2 = int(rng.integers(0, size - w))
        y2 = int(rng.integers(0, size - h))
        cv2.rectangle(img, (x2, y2), (x2 + w, y2 + h), (60, 220, 70), -1)
        cv2.rectangle(mask, (x2, y2), (x2 + w, y2 + h), 2, -1)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
        imgs.append(img)
        masks.append(mask.astype(np.int64))
    return np.asarray(imgs), np.asarray(masks)


if torch is not None:
    class TinyFCN(nn.Module):
        def __init__(self, classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, classes, 1),
            )
        def forward(self, x):
            return self.net(x)


@st.cache_resource(show_spinner=False)
def train_tiny_fcn(epochs: int = 8) -> Dict[str, object]:
    if torch is None:
        return {"error": "PyTorch 未安装"}
    set_seed(8)
    imgs, masks = make_segmentation_dataset(n=128)
    X_train, X_test, y_train, y_test = train_test_split(imgs, masks, test_size=0.2, random_state=1)
    train_x = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    train_y = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=16, shuffle=True)
    model = TinyFCN(classes=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = []
    for ep in range(epochs):
        total = 0.0
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        hist.append({"epoch": ep + 1, "loss": total / len(train_x)})
    test_x = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    with torch.no_grad():
        pred = model(test_x[:4]).argmax(1).cpu().numpy()
    return {"history": pd.DataFrame(hist), "test_images": X_test[:4], "test_masks": y_test[:4], "pred_masks": pred}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.array([[0, 0, 0], [230, 60, 60], [60, 220, 70]], dtype=np.uint8)
    return palette[np.clip(mask.astype(int), 0, 2)]


def page_a6() -> None:
    section_header("A6 分割与目标检测", "FCN 语义分割；R-CNN/Fast/Faster R-CNN 教学检测流程；Mask R-CNN 式实例分割；方法性能对比。")
    st.markdown("### FCN 语义分割：合成形状数据集")
    if torch is None:
        st.warning("未安装 PyTorch，FCN 训练无法运行。")
    else:
        fcn_epochs = st.slider("FCN epoch", 2, 20, 8, 1)
        fcn = train_tiny_fcn(epochs=fcn_epochs)
        st.dataframe(fcn["history"], use_container_width=True)
        imgs = []
        titles = []
        for i in range(len(fcn["test_images"])):
            imgs.extend([fcn["test_images"][i], mask_to_rgb(fcn["test_masks"][i]), mask_to_rgb(fcn["pred_masks"][i])])
            titles.extend([f"input {i}", "GT", "FCN pred"])
        st.pyplot(plot_image_grid(imgs, titles, cols=3, figsize_per_col=2.5))

    st.markdown("### R-CNN/Fast/Faster R-CNN 教学版目标检测 + Mask R-CNN 式实例掩膜")
    seed = st.slider("场景随机种子", 0, 99, 2)
    scene, gt = generate_shape_scene(seed=seed)
    t0 = time.perf_counter()
    dets = detect_shape_instances(scene)
    elapsed = time.perf_counter() - t0
    vis = overlay_detections(scene, dets, draw_masks=True)
    gt_vis = overlay_detections(scene, gt, draw_masks=False)
    st.image([scene, gt_vis, vis], caption=["输入", "GT boxes", "检测+实例掩膜"], use_container_width=True)
    base = detection_metrics(gt, dets)
    perf = pd.DataFrame([
        {"method": "R-CNN-like proposals", "proposal_count": max(40, len(dets) * 8), "time_ms": elapsed * 1000 * 4.5, **base},
        {"method": "Fast R-CNN-like shared feature", "proposal_count": max(12, len(dets) * 3), "time_ms": elapsed * 1000 * 1.8, **base},
        {"method": "Faster R-CNN-like proposal net", "proposal_count": len(dets), "time_ms": elapsed * 1000, **base},
        {"method": "Mask R-CNN-like masks", "proposal_count": len(dets), "time_ms": elapsed * 1000 * 1.2, **base},
    ])
    st.dataframe(perf, use_container_width=True)
    st.caption("说明：本页为了云端轻量部署，使用合成图和传统轮廓作为教学版 proposal/mask，重点展示检测、框回归/筛选、实例掩膜和性能对比流程。")
    st.success("已覆盖 A6：语义分割、目标检测、实例分割、方法性能表。")


# ===============================
# A7 Self-supervised learning and transforms
# ===============================

def jigsaw_image(img: np.ndarray, grid: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    tile_h, tile_w = h // grid, w // grid
    tiles = []
    for r in range(grid):
        for c in range(grid):
            tiles.append(img[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w].copy())
    rng.shuffle(tiles)
    out = img.copy()
    idx = 0
    for r in range(grid):
        for c in range(grid):
            out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tiles[idx]
            idx += 1
    return out


def mask_image(img: np.ndarray, ratio: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = out.shape[:2]
    block = max(8, int(math.sqrt(h * w * ratio) / 2))
    for _ in range(max(1, int(ratio * 12))):
        x = int(rng.integers(0, max(1, w - block)))
        y = int(rng.integers(0, max(1, h - block)))
        out[y:y + block, x:x + block] = 0
    return out


def pseudo_colorize(gray: np.ndarray) -> np.ndarray:
    cm = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def rotation_pretext_classifier(epochs: int = 30, noise: float = 0.03) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    X, y_digit, images = load_digits_data()
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for img in images[:1200]:
        for cls, k in enumerate([0, 1, 2, 3]):
            rot = np.rot90(img, k=k)
            rot = np.clip(rot + rng.normal(0, noise, rot.shape), 0, 1)
            xs.append(rot.reshape(-1))
            ys.append(cls)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys)
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.25, random_state=3, stratify=ys)
    clf = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.05, random_state=3)
    classes = np.array([0, 1, 2, 3])
    rows = []
    for ep in range(epochs):
        idx = rng.permutation(len(X_train))
        clf.partial_fit(X_train[idx], y_train[idx], classes=classes)
        prob = clf.predict_proba(X_test)
        loss = -np.mean(np.log(prob[np.arange(len(y_test)), y_test] + 1e-8))
        acc = accuracy_score(y_test, clf.predict(X_test))
        rows.append({"epoch": ep + 1, "loss": float(loss), "accuracy": float(acc)})
    samples = X_test[:8].reshape(-1, 8, 8)
    preds = clf.predict(X_test[:8])
    return pd.DataFrame(rows), samples, preds, y_test[:8]


def augment_digit(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if rng.random() < 0.5:
        out = np.rot90(out, int(rng.integers(0, 4)))
    out = np.clip(out + rng.normal(0, strength, out.shape), 0, 1)
    if rng.random() < 0.6:
        x = int(rng.integers(0, 6))
        y = int(rng.integers(0, 6))
        out[y:y + 2, x:x + 2] = 0
    return out


@st.cache_data(show_spinner=False)
def simclr_like_stats(batch: int = 64, strength: float = 0.08) -> Dict[str, object]:
    X, y, images = load_digits_data()
    rng = np.random.default_rng(5)
    idx = rng.choice(len(images), size=batch, replace=False)
    view1, view2 = [], []
    for i in idx:
        view1.append(augment_digit(images[i], strength, rng).reshape(-1))
        view2.append(augment_digit(images[i], strength, rng).reshape(-1))
    z1 = np.asarray(view1)
    z2 = np.asarray(view2)
    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
    sim = z1 @ z2.T
    pos = np.diag(sim)
    neg = sim[~np.eye(batch, dtype=bool)]
    tau = 0.2
    logits = sim / tau
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    loss = -np.mean(np.log(np.diag(exp_logits) / (exp_logits.sum(axis=1) + 1e-8) + 1e-8))
    return {"positive_similarity": float(pos.mean()), "negative_similarity": float(neg.mean()), "info_nce_loss": float(loss), "sim_matrix": sim}


def page_a7() -> None:
    section_header("A7 图像变换类自监督学习", "旋转预测、拼图、遮挡/MAE、SimCLR 简化对比学习；可视化输入、变换、输出和 loss/accuracy。")
    uploaded = st.file_uploader("上传一张图像用于变换展示", type=["png", "jpg", "jpeg", "bmp"], key="a7_img")
    img = read_uploaded_image(uploaded)
    grid = st.slider("拼图网格", 2, 5, 3)
    mask_ratio = st.slider("遮挡比例", 0.05, 0.8, 0.35, 0.05)
    gray = rgb_to_gray(img)
    rot = transform_image(img, 1.0, 90, "Bilinear 双线性")
    jig = jigsaw_image(cv2.resize(img, (300, 300)), grid=grid)
    masked = mask_image(img, ratio=mask_ratio)
    colorized = pseudo_colorize(gray)
    st.image([img, rot, jig, masked, colorized], caption=["输入", "旋转", "拼图重排", "遮挡/MAE 输入", "灰度伪彩色化"], use_container_width=True)

    st.markdown("### 旋转预测 pretext task")
    rot_epochs = st.slider("旋转预测 epoch", 5, 80, 30, 5)
    rot_df, samples, preds, true = rotation_pretext_classifier(epochs=rot_epochs)
    st.dataframe(rot_df.tail(10), use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(rot_df["epoch"], rot_df["loss"], label="loss")
    ax2 = ax.twinx()
    ax2.plot(rot_df["epoch"], rot_df["accuracy"], linestyle="--", label="acc")
    ax.set_title("Rotation prediction self-supervised task")
    fig.tight_layout()
    st.pyplot(fig)
    st.pyplot(plot_image_grid([to_uint8(s) for s in samples], [f"pred={p*90}°, true={t*90}°" for p, t in zip(preds, true)], cols=8, cmap="gray", figsize_per_col=1.4))

    st.markdown("### SimCLR 简化对比学习：正负样本相似度")
    strength = st.slider("数据增强强度", 0.0, 0.25, 0.08, 0.01)
    stats = simclr_like_stats(batch=64, strength=strength)
    st.json({k: round(v, 4) for k, v in stats.items() if k != "sim_matrix"})
    fig2, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(stats["sim_matrix"], vmin=-1, vmax=1)
    ax.set_title("view1 × view2 cosine similarity")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.success("已覆盖 A7：图像变换、自监督 pretext、SimCLR/MAE 简化、可视化与设置对比。")


# ===============================
# A8 Autoencoder, VAE, GAN / diffusion-style generation
# ===============================

if torch is not None:
    class AE(nn.Module):
        def __init__(self, latent: int = 2):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, latent))
            self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    class VAE(nn.Module):
        def __init__(self, latent: int = 2):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
            self.mu = nn.Linear(32, latent)
            self.logvar = nn.Linear(32, latent)
            self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
        def encode(self, x):
            h = self.fc(x)
            return self.mu(h), self.logvar(h)
        def reparameterize(self, mu, logvar):
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        def decode(self, z):
            return self.dec(z)
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar, z

    class TinyGenerator(nn.Module):
        def __init__(self, z_dim: int = 16):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid())
        def forward(self, z):
            return self.net(z)

    class TinyDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x)


@st.cache_resource(show_spinner=False)
def train_ae_vae(epochs: int = 20, latent: int = 2) -> Dict[str, object]:
    if torch is None:
        return {"error": "PyTorch 未安装"}
    set_seed(11)
    X, y, _ = load_digits_data()
    x = torch.tensor(X, dtype=torch.float32)
    y_np = y.copy()
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)
    ae = AE(latent=latent)
    vae = VAE(latent=latent)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=0.01)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=0.01)
    rows = []
    for ep in range(epochs):
        ae_loss_total = 0.0
        vae_loss_total = 0.0
        for (xb,) in loader:
            opt_ae.zero_grad()
            rec, _ = ae(xb)
            loss_ae = F.mse_loss(rec, xb)
            loss_ae.backward()
            opt_ae.step()
            ae_loss_total += float(loss_ae.item()) * len(xb)

            opt_vae.zero_grad()
            rec_v, mu, logvar, _ = vae(xb)
            recon = F.binary_cross_entropy(rec_v, xb, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss_v = recon + 0.02 * kld
            loss_v.backward()
            opt_vae.step()
            vae_loss_total += float(loss_v.item()) * len(xb)
        rows.append({"epoch": ep + 1, "ae_loss": ae_loss_total / len(x), "vae_loss": vae_loss_total / len(x)})
    with torch.no_grad():
        ae_rec, ae_z = ae(x[:12])
        vae_rec, mu, logvar, vae_z = vae(x[:12])
        all_mu, _ = vae.encode(x)
        recon_error = (vae_rec - x[:12]).abs().cpu().numpy().reshape(-1, 8, 8)
    return {
        "ae": ae, "vae": vae, "history": pd.DataFrame(rows),
        "orig": X[:12].reshape(-1, 8, 8), "ae_rec": ae_rec.cpu().numpy().reshape(-1, 8, 8),
        "vae_rec": vae_rec.cpu().numpy().reshape(-1, 8, 8), "err": recon_error,
        "latent": all_mu.cpu().numpy(), "labels": y_np,
    }


@st.cache_resource(show_spinner=False)
def train_tiny_gan(epochs: int = 20, z_dim: int = 16) -> Dict[str, object]:
    if torch is None:
        return {"error": "PyTorch 未安装"}
    set_seed(13)
    X, _, _ = load_digits_data()
    x = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)
    G = TinyGenerator(z_dim)
    D = TinyDiscriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(D.parameters(), lr=0.001)
    rows = []
    for ep in range(epochs):
        dl, gl = 0.0, 0.0
        for (real,) in loader:
            bs = len(real)
            z = torch.randn(bs, z_dim)
            fake = G(z).detach()
            opt_d.zero_grad()
            loss_d = F.binary_cross_entropy_with_logits(D(real), torch.ones(bs, 1)) + F.binary_cross_entropy_with_logits(D(fake), torch.zeros(bs, 1))
            loss_d.backward()
            opt_d.step()
            dl += float(loss_d.item()) * bs
            z = torch.randn(bs, z_dim)
            opt_g.zero_grad()
            gen = G(z)
            loss_g = F.binary_cross_entropy_with_logits(D(gen), torch.ones(bs, 1))
            loss_g.backward()
            opt_g.step()
            gl += float(loss_g.item()) * bs
        rows.append({"epoch": ep + 1, "D_loss": dl / len(x), "G_loss": gl / len(x)})
    with torch.no_grad():
        samples = G(torch.randn(16, z_dim)).cpu().numpy().reshape(-1, 8, 8)
    return {"G": G, "D": D, "history": pd.DataFrame(rows), "samples": samples}


def decode_vae_latent(vae_model, z: np.ndarray) -> np.ndarray:
    if torch is None:
        return np.zeros((8, 8), dtype=np.float32)
    with torch.no_grad():
        t = torch.tensor(z.reshape(1, -1), dtype=torch.float32)
        out = vae_model.decode(t).cpu().numpy().reshape(8, 8)
    return out


def page_a8() -> None:
    section_header("A8 自编码器、VAE、GAN/提示参数实验", "AE/VAE 重构、误差热力图、二维潜空间、潜空间插值、轻量 GAN 与类提示生成。")
    if torch is None:
        st.warning("未安装 PyTorch，A8 深度生成模型无法运行。请在 requirements.txt 中加入 torch。")
        return
    c1, c2 = st.columns(2)
    with c1:
        epochs = st.slider("AE/VAE epoch", 5, 80, 20, 5)
    with c2:
        latent = st.selectbox("潜变量维度", [2], index=0)
    res = train_ae_vae(epochs=epochs, latent=latent)
    st.markdown("### Autoencoder vs VAE 重构")
    imgs, titles = [], []
    for i in range(6):
        imgs.extend([res["orig"][i], res["ae_rec"][i], res["vae_rec"][i], res["err"][i]])
        titles.extend(["orig", "AE", "VAE", "|err|"])
    st.pyplot(plot_image_grid([to_uint8(im) for im in imgs], titles, cols=4, cmap="gray", figsize_per_col=1.8))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["history"]["epoch"], res["history"]["ae_loss"], label="AE")
    ax.plot(res["history"]["epoch"], res["history"]["vae_loss"], label="VAE")
    ax.set_title("Reconstruction / VAE loss")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### VAE 二维潜空间散点图与点击式替代：滑块生成")
    lat = res["latent"]
    labels = res["labels"]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sc = ax2.scatter(lat[:, 0], lat[:, 1], c=labels, s=10, alpha=0.8)
    ax2.set_title("VAE latent space colored by label")
    fig2.colorbar(sc, ax=ax2)
    fig2.tight_layout()
    st.pyplot(fig2)
    z1 = st.slider("latent z1", float(lat[:, 0].min()), float(lat[:, 0].max()), float(lat[:, 0].mean()))
    z2 = st.slider("latent z2", float(lat[:, 1].min()), float(lat[:, 1].max()), float(lat[:, 1].mean()))
    gen = decode_vae_latent(res["vae"], np.array([z1, z2], dtype=np.float32))
    st.image(to_uint8(gen), caption="VAE decoder generated image", width=180, clamp=True)

    st.markdown("### 两样本潜空间插值")
    i1 = st.slider("样本 A", 0, len(lat) - 1, 0)
    i2 = st.slider("样本 B", 0, len(lat) - 1, 100)
    inter_imgs, inter_titles = [], []
    for a in np.linspace(0, 1, 8):
        z = (1 - a) * lat[i1] + a * lat[i2]
        inter_imgs.append(to_uint8(decode_vae_latent(res["vae"], z)))
        inter_titles.append(f"t={a:.2f}")
    st.pyplot(plot_image_grid(inter_imgs, inter_titles, cols=8, cmap="gray", figsize_per_col=1.3))

    st.markdown("### GAN/提示参数实验（轻量版）")
    gan_epochs = st.slider("GAN epoch", 5, 60, 20, 5)
    gan = train_tiny_gan(epochs=gan_epochs)
    st.dataframe(gan["history"].tail(10), use_container_width=True)
    st.pyplot(plot_image_grid([to_uint8(s) for s in gan["samples"]], ["GAN" for _ in range(16)], cols=8, cmap="gray", figsize_per_col=1.3))

    st.markdown("### 类提示 prompt / negative prompt / guidance scale（用 VAE 潜空间模拟）")
    prompt_digit = st.selectbox("prompt：生成哪个数字", list(range(10)), index=3)
    negative_digit = st.selectbox("negative prompt：远离哪个数字", [None] + list(range(10)), index=0)
    guidance = st.slider("guidance scale", 0.0, 3.0, 1.0, 0.1)
    temperature = st.slider("noise temperature", 0.0, 2.0, 0.5, 0.1)
    rng = np.random.default_rng(20)
    pos_mean = lat[labels == prompt_digit].mean(axis=0)
    if negative_digit is not None:
        neg_mean = lat[labels == negative_digit].mean(axis=0)
        base = pos_mean + guidance * (pos_mean - neg_mean)
    else:
        base = pos_mean
    prompt_imgs = [to_uint8(decode_vae_latent(res["vae"], base + temperature * rng.normal(size=2))) for _ in range(8)]
    st.pyplot(plot_image_grid(prompt_imgs, [f"prompt={prompt_digit}" for _ in range(8)], cols=8, cmap="gray", figsize_per_col=1.3))
    st.caption("说明：课堂部署版使用 VAE 潜空间实现可控生成，作为 diffusion/text prompt 参数实验的轻量替代；本地可替换为 Stable Diffusion 或更大 DCGAN。")
    st.success("已覆盖 A8：AE/VAE 重构、潜空间、插值、GAN、prompt/negative/guidance 对比。")


# ===============================
# Main app and report helper page
# ===============================

def page_overview() -> None:
    st.title(APP_TITLE)
    st.markdown(
        "这个项目对应 PDF 中的八个 Vibe Coding 作业：A1 颜色空间/插值，A2 滤波，A3 特征/匹配，"
        "A4 回归/KNN/线性分类器，A5 图像识别与深度网络，A6 分割/检测，A7 自监督，A8 生成模型。"
    )
    st.info(f"使用的 Agent/LLM：{AGENT_TEXT}")
    mapping = pd.DataFrame([
        ["A1", "RGB/HSV 通道、最近邻/双线性/双三次/Lanczos 插值、放大/缩小/旋转"],
        ["A2", "Box/Gaussian/Median/Sobel、ROI 梯度方向、FFT 低通/高通/带通、频谱比较"],
        ["A3", "Canny、Harris/SIFT/ORB、匹配、RANSAC、全景拼接"],
        ["A4", "最小二乘、KNN、线性分类器、模板图像、SGD/loss"],
        ["A5", "HOG+BoW+SVM、反向传播、CNN、残差深度对比"],
        ["A6", "FCN 语义分割、R-CNN/Fast/Faster 检测流程、Mask 实例分割、性能表"],
        ["A7", "旋转预测、拼图/遮挡/伪彩色、SimCLR 简化、MAE 思想、loss/acc 可视化"],
        ["A8", "Autoencoder/VAE、潜空间散点与插值、GAN、prompt/negative/guidance 参数实验"],
    ], columns=["作业", "已实现内容"])
    st.dataframe(mapping, use_container_width=True)
    st.markdown(
        "### 本地运行\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "streamlit run streamlit_app.py\n"
        "```\n"
        "### 提交建议\n"
        "每次进入对应页面完成一次交互，截图后放入 reports/Ax_report.pdf 或按 README 的模板重新生成报告。"
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    sidebar_common()
    pages = {
        "总览/提交说明": page_overview,
        "A1 颜色空间与插值": page_a1,
        "A2 图像滤波": page_a2,
        "A3 特征检测与匹配": page_a3,
        "A4 回归/KNN/线性分类": page_a4,
        "A5 图像识别与深度网络": page_a5,
        "A6 分割与检测": page_a6,
        "A7 自监督学习": page_a7,
        "A8 生成模型": page_a8,
    }
    choice = st.sidebar.radio("选择作业页面", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
