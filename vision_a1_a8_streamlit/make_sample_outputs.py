"""Generate static example figures used in the PDF reports."""
import sys, types, importlib.util
from pathlib import Path

class ST(types.SimpleNamespace):
    def cache_data(self, *args, **kwargs):
        def dec(f): return f
        if args and callable(args[0]): return args[0]
        return dec
    def cache_resource(self, *args, **kwargs):
        def dec(f): return f
        if args and callable(args[0]): return args[0]
        return dec
    def __getattr__(self, name):
        def dummy(*args, **kwargs): return None
        return dummy
sys.modules['streamlit'] = ST()

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('app', ROOT / 'streamlit_app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

OUT = ROOT / 'sample_outputs'
OUT.mkdir(exist_ok=True)

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

img = app.demo_image()
gray = app.rgb_to_gray(img)

# A1
images, titles = app.color_channel_images(img)
out = app.transform_image(img, 1.25, 25, 'Bicubic 双三次')
save_fig(app.plot_image_grid(images + [out], titles + ['Bicubic transform'], cols=4), 'A1_sample.png')

# A2
filtered = app.apply_spatial_filter(gray, 'Gaussian 高斯滤波', 5, 1.2)
mag, edges = app.draw_canny_comparison(gray, 60, 140)
spectrum, mask, freq_out = app.fft_filter(gray, '低通 Low-pass', 35)
save_fig(app.plot_image_grid([gray, filtered, mag, edges, spectrum, mask, freq_out], ['gray','Gaussian','gradient','Canny','FFT spectrum','LP mask','LP output'], cols=4), 'A2_sample.png')

# A3
kimg, _, _ = app.detect_keypoints(img, 'SIFT')
img1, img2 = app.make_matching_pair(img)
match_img, metrics = app.match_two_images(img1, img2, 'SIFT')
save_fig(app.plot_image_grid([app.to_uint8(edges), kimg, match_img], ['Canny edges','SIFT keypoints',f'Matches/RANSAC {metrics}'], cols=1, figsize_per_col=6), 'A3_sample.png')

# A4
fig, _ = app.linear_regression_demo(80, 0.8, 42)
save_fig(fig, 'A4_regression.png')
results, weights, templates = app.run_knn_linear_demo()
save_fig(app.plot_image_grid([app.to_uint8(t) for t in templates] + [app.to_uint8(w) for w in weights], [f'mean {i}' for i in range(10)] + [f'w {i}' for i in range(10)], cols=10, cmap='gray', figsize_per_col=1.4), 'A4_templates.png')
losses, learned = app.softmax_sgd_digits(epochs=80, lr=0.8)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,3.5))
ax.plot(losses['epoch'], losses['loss'], label='loss')
ax2 = ax.twinx(); ax2.plot(losses['epoch'], losses['test_accuracy'], linestyle='--', label='acc')
ax.set_title('A4 SGD loss and accuracy')
fig.tight_layout(); save_fig(fig, 'A4_loss.png')

# A5
bow = app.train_bow_svm(16)
X, y, imgs = app.load_digits_data()
fd, hog_img = app.hog(imgs[0], orientations=8, pixels_per_cell=(4,4), cells_per_block=(1,1), visualize=True)
bp = app.train_mlp_backprop(hidden=24, epochs=300)
fig_bp = app.mlp_decision_boundary(bp)
save_fig(app.plot_image_grid([imgs[0], hog_img] + [app.to_uint8(c) for c in bow['centers'][:8]], ['sample','HOG']+[f'word {i}' for i in range(8)], cols=5, cmap='gray', figsize_per_col=2), 'A5_hog_bow.png')
save_fig(fig_bp, 'A5_backprop.png')
if app.torch is not None:
    cnn = app.train_tiny_cnn(epochs=3)
    res = app.compare_resnet_depths(epochs=5)
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.plot(cnn['history']['epoch'], cnn['history']['loss'], label='CNN loss')
    ax2 = ax.twinx(); ax2.plot(cnn['history']['epoch'], cnn['history']['test_accuracy'], linestyle='--', label='acc')
    ax.set_title('Tiny CNN training')
    fig.tight_layout(); save_fig(fig, 'A5_cnn.png')
    fig, ax = plt.subplots(figsize=(6,3.5))
    for name, sub in res.groupby('type'):
        ax.plot(sub['depth'], sub['test_accuracy'], marker='o', label=name)
    ax.set_title('Plain vs Residual depth')
    ax.legend(); fig.tight_layout(); save_fig(fig, 'A5_resnet.png')

# A6
scene, gt = app.generate_shape_scene(seed=2)
dets = app.detect_shape_instances(scene)
vis = app.overlay_detections(scene, dets, draw_masks=True)
gt_vis = app.overlay_detections(scene, gt, draw_masks=False)
save_fig(app.plot_image_grid([scene, gt_vis, vis], ['input','GT boxes','Mask detection'], cols=3), 'A6_detection.png')
if app.torch is not None:
    fcn = app.train_tiny_fcn(epochs=3)
    imgs6=[]; titles6=[]
    for i in range(2):
        imgs6 += [fcn['test_images'][i], app.mask_to_rgb(fcn['test_masks'][i]), app.mask_to_rgb(fcn['pred_masks'][i])]
        titles6 += ['input','GT','FCN pred']
    save_fig(app.plot_image_grid(imgs6,titles6,cols=3,figsize_per_col=2.2),'A6_fcn.png')

# A7
jig = app.jigsaw_image(__import__('cv2').resize(img, (300,300)), grid=3)
masked = app.mask_image(img, ratio=0.35)
colorized = app.pseudo_colorize(gray)
rot = app.transform_image(img, 1, 90, 'Bilinear 双线性')
save_fig(app.plot_image_grid([img, rot, jig, masked, colorized], ['input','rotation','jigsaw','mask/MAE','pseudo color'], cols=5), 'A7_transforms.png')
rot_df, samples, preds, true = app.rotation_pretext_classifier(epochs=20)
fig, ax = plt.subplots(figsize=(6,3.5)); ax.plot(rot_df['epoch'], rot_df['loss'], label='loss'); ax2=ax.twinx(); ax2.plot(rot_df['epoch'], rot_df['accuracy'], linestyle='--', label='acc'); ax.set_title('Rotation prediction'); fig.tight_layout(); save_fig(fig, 'A7_rotation_loss.png')
stats = app.simclr_like_stats(batch=64, strength=0.08)
fig, ax = plt.subplots(figsize=(4,3.5)); ax.imshow(stats['sim_matrix'], vmin=-1, vmax=1); ax.set_title('SimCLR-like similarity'); fig.tight_layout(); save_fig(fig,'A7_simclr.png')

# A8
if app.torch is not None:
    ae = app.train_ae_vae(epochs=10)
    imgs8=[]; titles8=[]
    for i in range(4):
        imgs8 += [ae['orig'][i], ae['ae_rec'][i], ae['vae_rec'][i], ae['err'][i]]
        titles8 += ['orig','AE','VAE','err']
    save_fig(app.plot_image_grid([app.to_uint8(x) for x in imgs8], titles8, cols=4, cmap='gray', figsize_per_col=1.8), 'A8_recon.png')
    lat=ae['latent']; labels=ae['labels']
    fig, ax = plt.subplots(figsize=(5,4)); sc=ax.scatter(lat[:,0], lat[:,1], c=labels, s=8, alpha=0.8); ax.set_title('VAE latent'); fig.colorbar(sc, ax=ax); fig.tight_layout(); save_fig(fig,'A8_latent.png')
    gan = app.train_tiny_gan(epochs=10)
    save_fig(app.plot_image_grid([app.to_uint8(s) for s in gan['samples']], ['GAN' for _ in range(16)], cols=8, cmap='gray', figsize_per_col=1.2), 'A8_gan.png')

print('sample outputs generated in', OUT)
