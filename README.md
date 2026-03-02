# Sci-Dreamer

**孩子的发明，AI 让它成真**

首个开源儿童科学想象力可视化平台。填写一张发明卡，Sci-Dreamer 将孩子的"童言童语"转化为电影级科幻海报。

由 天与ARTECH工作室 出品 · SIGGRAPH 2026 Art Paper 实体展示平台

---

## 技术栈

- Backend: Flask + Gunicorn
- Image Generation: Google Imagen 4 (via Gemini API)
- Prompt Generation: OpenAI GPT-4
- Poster Composition: Pillow

## 本地运行

```bash
pip install -r requirements.txt
python app.py
```

## 环境变量

部署时需要设置：

- `OPENAI_API_KEY` — OpenAI API Key
- `GEMINI_API_KEY` — Google Gemini API Key
