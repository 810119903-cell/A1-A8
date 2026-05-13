from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
SAMPLES = ROOT / 'sample_outputs'
REPORTS.mkdir(exist_ok=True)

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name='CJKTitle', parent=styles['Title'], fontName='STSong-Light', fontSize=20, leading=28,
    alignment=TA_CENTER, spaceAfter=12
))
styles.add(ParagraphStyle(
    name='CJKHeading1', parent=styles['Heading1'], fontName='STSong-Light', fontSize=15, leading=21,
    spaceBefore=8, spaceAfter=6
))
styles.add(ParagraphStyle(
    name='CJKBody', parent=styles['BodyText'], fontName='STSong-Light', fontSize=10.5, leading=17,
    alignment=TA_LEFT, spaceAfter=4
))
styles.add(ParagraphStyle(
    name='CJKSmall', parent=styles['BodyText'], fontName='STSong-Light', fontSize=9, leading=14,
    textColor=colors.darkgrey, spaceAfter=4
))
styles.add(ParagraphStyle(
    name='CJKMono', parent=styles['BodyText'], fontName='STSong-Light', fontSize=9.5, leading=15,
    backColor=colors.whitesmoke, borderColor=colors.lightgrey, borderWidth=0.5, borderPadding=5, spaceAfter=6
))

AGENT = 'GPT-5.5 Pro + Python/OpenCV/scikit-learn/PyTorch/Streamlit'

DATA = {
'A1': {
 'title': 'A1 图像颜色空间、图像插值',
 'requirements': ['实现 RGB、HSV 颜色空间相关转换，输入图像并输出不同颜色空间每个通道图像。', '实现最近邻、双线性、双三次、Lanczos 插值，用于放大、缩小和旋转。', '使用 Python + OpenCV + Streamlit 做交互式应用。'],
 'implementation': ['页面 page_a1() 提供图像上传、RGB/HSV 通道可视化、缩放倍数、旋转角度和插值方法选择。', '核心函数 color_channel_images() 与 transform_image() 分别完成通道拆分和插值变换。'],
 'prompt': '请使用 Python、OpenCV 和 Streamlit 完成 A1：输入图像后展示 RGB/HSV 每个通道；提供最近邻、双线性、双三次、Lanczos 插值选择；支持放大、缩小、旋转；输出可下载结果图，并写出实验总结。',
 'steps': ['运行 streamlit run streamlit_app.py，左侧选择“A1 颜色空间与插值”。', '上传图像或使用默认图；观察 R/G/B/H/S/V 通道图。', '调整缩放倍数、旋转角度和插值算法，比较边缘锯齿、模糊程度和细节保留。'],
 'images': ['A1_sample.png'],
 'summary': '最近邻速度快但锯齿明显；双线性较平滑；双三次和 Lanczos 对放大细节更友好。HSV 的 H 通道突出色调，S 通道表示饱和度，V 通道表示亮度。'
},
'A2': {
 'title': 'A2 图像滤波',
 'requirements': ['实现常见空间图像滤波器 Box/Gaussian/Median/Sobel 并比较。', '演示局部区域梯度方向计算。', '实现傅里叶变换/反变换的频率域滤波，绘制频谱并比较旋转缩放下的谱图变化。'],
 'implementation': ['page_a2() 包含空间滤波、ROI 梯度矢量图、FFT 低通/高通/带通滤波。', '核心函数 apply_spatial_filter()、gradient_roi_figure()、fft_filter()。'],
 'prompt': '请用 Python、OpenCV、NumPy、Streamlit 完成 A2：实现 Box/Gaussian/Median/Sobel 空间滤波；允许框选 ROI 并计算梯度方向；用 FFT 实现低通、高通、带通滤波并显示频谱、掩膜和反变换图。',
 'steps': ['进入“A2 图像滤波”页面。', '选择不同滤波器和核大小，记录平滑、去噪和边缘增强效果。', '用 ROI 滑块选择局部区域，观察梯度方向箭头。', '切换低通/高通/带通滤波，比较频谱和反变换结果。'],
 'images': ['A2_sample.png'],
 'summary': '均值与高斯滤波能平滑噪声，中值滤波对椒盐噪声更稳健，Sobel/Laplacian 强调边缘。低通保留整体轮廓，高通保留边缘纹理，频谱会随图像旋转而产生对应旋转。'
},
'A3': {
 'title': 'A3 图像特征检测与匹配',
 'requirements': ['实现 Canny 边缘检测，并展示非最大值抑制前后对比。', '实现 Harris、SIFT 特征点检测并可视化。', '实现两幅图像的特征匹配、描述、初始匹配、RANSAC、变换和对齐。', '实现多图像全景拼接并比较 blending 区别。'],
 'implementation': ['page_a3() 包含 Canny、Harris/SIFT/ORB、双图匹配与 RANSAC、OpenCV Stitcher 全景拼接。', '核心函数 draw_canny_comparison()、detect_keypoints()、match_two_images()、stitch_images()。'],
 'prompt': '请用 OpenCV + Streamlit 完成 A3：实现 Canny 边缘检测与梯度候选对比；实现 Harris/SIFT/ORB 特征点可视化；对两张图做特征匹配、RANSAC 内点筛选和匹配图展示；对多张同场景图片尝试全景拼接。',
 'steps': ['进入“A3 特征检测与匹配”页面。', '调节 Canny 双阈值，比较梯度候选图和最终边缘。', '选择 Harris/SIFT/ORB，观察特征点位置与尺度。', '上传两张相似图片或使用自动生成的旋转平移图，查看匹配数和 RANSAC 内点数。', '上传多张重叠图片，生成全景结果。'],
 'images': ['A3_sample.png'],
 'summary': 'Canny 的双阈值影响边缘连续性；SIFT/ORB 能提供局部描述子，RANSAC 可剔除错误匹配并估计单应变换。全景拼接依赖足够重叠区域和稳定特征。'
},
'A4': {
 'title': 'A4 回归、KNN 与线性分类器',
 'requirements': ['实现 Least Squares Linear Regression 示例。', '实现 KNN 与线性分类器图像分类。', '展示模板图像、不同 k 的 KNN 对比、SGD/动量更新思想、梯度下降和 loss 计算过程。'],
 'implementation': ['page_a4() 包含最小二乘拟合、digits 图像数据上的 KNN/Logistic 分类、均值模板与线性权重模板、Softmax SGD loss 曲线。', '核心函数 linear_regression_demo()、run_knn_linear_demo()、softmax_sgd_digits()。'],
 'prompt': '请使用 Python、scikit-learn、NumPy、Streamlit 完成 A4：实现最小二乘线性回归；用图像数据比较 KNN 不同 k 与线性分类器；可视化每类模板和线性权重；手写 softmax SGD 训练过程并显示 loss/accuracy 曲线。',
 'steps': ['进入“A4 回归/KNN/线性分类”页面。', '调整线性回归噪声和样本数，观察拟合线与 MSE。', '查看 KNN 不同 k 与线性分类器准确率表。', '观察类别均值模板和线性权重模板。', '调节 SGD epoch/学习率，观察 loss 与 accuracy。'],
 'images': ['A4_regression.png', 'A4_templates.png', 'A4_loss.png'],
 'summary': '最小二乘通过正规方程/最小化平方误差得到参数。KNN 依赖样本邻域，k 越大越平滑；线性分类器学习到每类模板权重。SGD 的 loss 曲线展示了迭代优化过程。'
},
'A5': {
 'title': 'A5 图像识别与深度网络',
 'requirements': ['实现 HOG + Bag of Words + SVM 图像识别/分类示例。', '实现神经网络反向传播演示。', '实现 CNN 模型训练和测试。', '对比不同深度的预训练/残差网络 ResNet 性能。'],
 'implementation': ['page_a5() 包含 HOG+BoW+SVM、两层 MLP 手写反向传播、Tiny CNN、Plain MLP 与 Residual MLP 深度对比。', '核心函数 train_bow_svm()、train_mlp_backprop()、train_tiny_cnn()、compare_resnet_depths()。'],
 'prompt': '请完成 A5：用 HOG 特征、视觉词袋和 SVM 做图像分类；用 NumPy 手写两层神经网络反向传播并展示决策边界和 loss；用 PyTorch 训练一个轻量 CNN；比较普通深层网络和残差连接网络在图像分类上的性能差异。',
 'steps': ['进入“A5 图像识别与深度网络”页面。', '设置视觉词数量，查看 HOG、视觉词和 SVM 准确率。', '调整反向传播隐藏层与 epoch，观察决策边界。', '运行 Tiny CNN 并记录 loss/accuracy。', '查看 Plain 与 Residual 网络深度对比图。'],
 'images': ['A5_hog_bow.png', 'A5_backprop.png', 'A5_cnn.png', 'A5_resnet.png'],
 'summary': 'HOG 描述边缘方向，BoW 把局部视觉词统计成直方图，SVM 完成分类。反向传播通过链式法则优化网络参数。残差连接能缓解深层普通网络训练退化。'
},
'A6': {
 'title': 'A6 语义分割、目标检测与实例分割',
 'requirements': ['实现 FCN 语义分割示例。', '实现 R-CNN/Fast/Faster R-CNN 目标检测示例。', '实现 Mask R-CNN 图像实例分割示例。', '对比不同方法性能。'],
 'implementation': ['page_a6() 使用合成形状数据训练 Tiny FCN 分割器，并用轮廓 proposal 展示 R-CNN/Fast/Faster R-CNN 教学流程和 Mask R-CNN 式实例掩膜。', '核心函数 train_tiny_fcn()、generate_shape_scene()、detect_shape_instances()、detection_metrics()。'],
 'prompt': '请用 Python、OpenCV、PyTorch、Streamlit 完成 A6：构造一个轻量 FCN 做语义分割；展示 R-CNN/Fast/Faster R-CNN 的 proposal、分类、框和性能对比流程；输出 Mask R-CNN 式实例 mask，并用表格比较耗时、proposal 数和 IoU。',
 'steps': ['进入“A6 分割与检测”页面。', '运行 FCN 训练，查看输入、GT mask、预测 mask。', '调整随机种子生成不同目标检测场景。', '查看检测框、实例掩膜、proposal 数、耗时和 IoU 表。'],
 'images': ['A6_fcn.png', 'A6_detection.png'],
 'summary': 'FCN 输出像素级类别；R-CNN 系列从区域 proposal 到共享特征再到 proposal 网络逐步提高效率；Mask R-CNN 在检测框基础上增加实例掩膜分支。本项目采用轻量合成数据保证云端可部署。'
},
'A7': {
 'title': 'A7 图像变换类自监督学习',
 'requirements': ['实现图像变换类自监督示例，如旋转预测、拼图重排、图像补全或颜色化。', '实现 MAE 或 SimCLR 简化示例。', '可视化输入、变换/遮挡后的图像、模型输出、loss 或准确率变化。', '对比不同设置效果。'],
 'implementation': ['page_a7() 包含旋转、拼图、遮挡、伪彩色化，旋转预测 pretext task，SimCLR 简化正负样本相似度和 InfoNCE loss。', '核心函数 jigsaw_image()、mask_image()、rotation_pretext_classifier()、simclr_like_stats()。'],
 'prompt': '请完成 A7：用图像旋转预测作为自监督 pretext；实现拼图重排、遮挡/MAE、颜色化展示；实现 SimCLR 简化版正负样本对比，输出相似度矩阵、InfoNCE loss，并比较不同增强强度。',
 'steps': ['进入“A7 自监督学习”页面。', '上传图像或使用默认图，观察旋转、拼图、遮挡和颜色化。', '运行旋转预测，查看 loss/accuracy 曲线和预测角度。', '调节增强强度，观察 SimCLR 正样本和负样本相似度变化。'],
 'images': ['A7_transforms.png', 'A7_rotation_loss.png', 'A7_simclr.png'],
 'summary': '自监督学习通过人为构造标签训练表征。旋转预测要求模型理解图像结构；SimCLR 强化同一图像不同增强视图的一致性；MAE 思想通过遮挡重建学习上下文。'
},
'A8': {
 'title': 'A8 自编码器、VAE、GAN 与提示参数实验',
 'requirements': ['实现自编码器与 VAE 重构对比，展示重构结果、误差热力图和 loss 曲线。', '提供 VAE 二维潜空间散点图，支持潜空间生成与插值。', '实现 GAN/扩散模型与文本提示参数实验，比较 prompt、negative prompt、guidance scale 等参数影响。'],
 'implementation': ['page_a8() 训练 AE/VAE，展示重构、误差、二维 latent、潜空间插值；训练轻量 GAN；用 VAE 潜空间模拟 prompt/negative/guidance 可控生成。', '核心函数 train_ae_vae()、decode_vae_latent()、train_tiny_gan()。'],
 'prompt': '请用 PyTorch + Streamlit 完成 A8：在 digits/MNIST 类图像上训练 Autoencoder 和 VAE，展示重构、误差热力图、loss 曲线、二维潜空间和插值；训练轻量 GAN；实现 prompt、negative prompt、guidance scale 和 noise 参数对生成图像的影响对比。',
 'steps': ['进入“A8 生成模型”页面。', '运行 AE/VAE 训练，查看原图、AE 重构、VAE 重构和误差。', '在二维潜空间散点图中观察类别分布，用 z1/z2 滑块生成图像。', '选择两个样本进行潜空间插值。', '运行轻量 GAN，并比较 prompt/negative/guidance 参数生成结果。'],
 'images': ['A8_recon.png', 'A8_latent.png', 'A8_gan.png'],
 'summary': 'AE 学习确定性压缩重构，VAE 学习连续概率潜空间，因而更适合采样和插值。GAN 通过生成器/判别器对抗训练生成样本。prompt/negative/guidance 可理解为在潜空间中靠近目标语义、远离负语义并控制生成强度。'
}
}

def P(text, style='CJKBody'):
    return Paragraph(escape(text).replace('\n', '<br/>'), styles[style])

def add_bullets(story, items):
    for item in items:
        story.append(P('- ' + item))


def add_image(story, filename, caption):
    path = SAMPLES / filename
    if not path.exists():
        story.append(P(f'示例图缺失：{filename}', 'CJKSmall'))
        return
    with PILImage.open(path) as im:
        w, h = im.size
    max_w = 16.0 * cm
    max_h = 9.0 * cm
    scale = min(max_w / w, max_h / h, 1.0)
    story.append(Image(str(path), width=w * scale, height=h * scale))
    story.append(P(caption, 'CJKSmall'))
    story.append(Spacer(1, 0.2 * cm))


def make_pdf(key, data):
    path = REPORTS / f'{key}_report.pdf'
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    story.append(P(f'{key} 实验报告：{data["title"]}', 'CJKTitle'))
    story.append(P(f'使用的 Agent/LLM：{AGENT}', 'CJKSmall'))
    story.append(P('一、作业要求', 'CJKHeading1'))
    add_bullets(story, data['requirements'])
    story.append(P('二、实现说明', 'CJKHeading1'))
    add_bullets(story, data['implementation'])
    story.append(P('三、Prompt（纯文本）', 'CJKHeading1'))
    story.append(P(data['prompt'], 'CJKMono'))
    story.append(P('四、测试步骤', 'CJKHeading1'))
    add_bullets(story, data['steps'])
    story.append(P('五、测试截图/输出示例', 'CJKHeading1'))
    for im in data['images']:
        add_image(story, im, f'图：{im}')
    story.append(P('六、实验小结', 'CJKHeading1'))
    story.append(P(data['summary']))
    story.append(P('七、核心源码位置', 'CJKHeading1'))
    story.append(P(f'streamlit_app.py 中的 page_{key.lower()}() 及其调用的辅助函数。'))
    story.append(P('提交时可同时附上 Streamlit Cloud URL 与 GitHub 仓库链接。', 'CJKSmall'))
    doc.build(story)
    return path


def make_md(key, data):
    path = REPORTS / f'{key}_report.md'
    lines = []
    lines.append(f'# {key} 实验报告：{data["title"]}')
    lines.append(f'使用的 Agent/LLM：{AGENT}')
    lines.append('\n## 一、作业要求')
    lines += [f'- {x}' for x in data['requirements']]
    lines.append('\n## 二、实现说明')
    lines += [f'- {x}' for x in data['implementation']]
    lines.append('\n## 三、Prompt（纯文本）')
    lines.append(data['prompt'])
    lines.append('\n## 四、测试步骤')
    lines += [f'- {x}' for x in data['steps']]
    lines.append('\n## 五、测试截图/输出示例')
    for im in data['images']:
        lines.append(f'![{im}](../sample_outputs/{im})')
    lines.append('\n## 六、实验小结')
    lines.append(data['summary'])
    lines.append('\n## 七、核心源码位置')
    lines.append(f'`streamlit_app.py` 中的 `page_{key.lower()}()` 及其调用的辅助函数。')
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

if __name__ == '__main__':
    made = []
    for key, data in DATA.items():
        made.append(make_pdf(key, data))
        make_md(key, data)
    print('Generated reports:')
    for p in made:
        print(p)
