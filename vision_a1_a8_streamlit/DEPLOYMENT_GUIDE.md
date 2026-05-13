# GitHub + Streamlit 提前部署教学清单

## 第 0 步：确认你已经有这些文件

- `streamlit_app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `reports/` 报告文件夹
- `README.md`

## 第 1 步：本地先跑通

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

逐页检查 A1-A8，尤其是 A5-A8：把训练 epoch 调小，确认不会卡住。

## 第 2 步：上传到 GitHub

GitHub 网页端创建新仓库后，本地执行：

```bash
git init
git add .
git commit -m "A1-A8 streamlit app"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

## 第 3 步：部署到 Streamlit Cloud

- Create app
- Repository：你的 GitHub 仓库
- Branch：`main`
- Main file path：`streamlit_app.py`
- Python version：建议 `3.12`；如果 PyTorch wheel 有兼容问题，改用 `3.11`
- Deploy

## 第 4 步：更新代码后重新部署

只要改代码后 push 到 GitHub，Streamlit Cloud 会自动检测并更新：

```bash
git add .
git commit -m "update app"
git push
```

如果改了 `requirements.txt`，云端会重新安装依赖，耗时会更久。

## 第 5 步：提交作业

把以下内容交给老师：

1. Streamlit URL。
2. 每个作业的 PDF 报告。
3. `streamlit_app.py` 或 GitHub 仓库链接。
