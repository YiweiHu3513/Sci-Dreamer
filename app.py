# -*- coding: utf-8 -*-
import os
import io
import base64
import json
import re
import datetime
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from google import genai as google_genai
from google.genai import types as google_types

app = Flask(__name__, static_folder='static')

# Clients — API keys from environment variables
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
gemini_client = google_genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

ASSETS_DIR   = os.path.join(os.path.dirname(__file__), 'assets')
RECORDS_DIR  = os.path.join(os.path.dirname(__file__), 'records')
os.makedirs(RECORDS_DIR, exist_ok=True)

# Font paths — use system NotoSansCJK for reliable CJK rendering
FONT_CJK_BLACK = os.path.join(ASSETS_DIR, 'NotoSansCJK-Black.ttc')
FONT_EN_BOLD   = os.path.join(ASSETS_DIR, 'NotoSans-Bold.ttf')
FONT_EN_REG    = os.path.join(ASSETS_DIR, 'NotoSans-Regular.ttf')

# Logo
LOGO_PATH = os.path.join(ASSETS_DIR, 'logo_light.png')

# Pre-rendered text layer PNGs (generated once at startup)
TEXT_LAYERS = {
    'zh': {
        'tagline': os.path.join(ASSETS_DIR, 'tagline_zh.png'),
        'credit':  os.path.join(ASSETS_DIR, 'credit_zh.png'),
        'name_font': FONT_CJK_BLACK,
        'inv_font':  FONT_CJK_BLACK,
    },
    'en': {
        'tagline': os.path.join(ASSETS_DIR, 'tagline_en.png'),
        'credit':  os.path.join(ASSETS_DIR, 'credit_en.png'),
        'name_font': FONT_EN_BOLD,
        'inv_font':  FONT_EN_BOLD,
    },
    'tw': {
        'tagline': os.path.join(ASSETS_DIR, 'tagline_tw.png'),
        'credit':  os.path.join(ASSETS_DIR, 'credit_tw.png'),
        'name_font': FONT_CJK_BLACK,
        'inv_font':  FONT_CJK_BLACK,
    },
}

# ── Prompt generation ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Sci-Dreamer's Prompt Builder for children's speculative invention cards about nuclear fusion.

Convert a child's invention description into a high-quality image generation prompt for Imagen 4.

Follow this Chain-of-Thought:
1. Function Recognition: identify the core invention concept
2. Material Inference: translate childlike metaphors into physical materials
   (e.g. "glows like firefly" → "bioluminescent neon tubing, Cherenkov radiation glow")
3. Parameter Generation: add photorealistic rendering parameters

Output ONLY a JSON object:
{
  "prompt": "detailed English prompt, starting with main subject, including materials, lighting, style. MUST end with: 16:9 aspect ratio, 1920x1080 resolution, no text, no watermark, no people, cinematic composition",
  "style_tags": ["sci-fi", "industrial", "speculative design"]
}

Rules:
- Prompt must be in English
- Include: volumetric lighting, photorealistic, 8K, hyperrealistic, sci-fi industrial aesthetic
- Preserve child's core concept but elevate to speculative design blueprint
- Background relates to usage scenario
- Always include: detailed mechanical design, speculative industrial aesthetic
- Keep prompt under 220 words
- NEVER include text, letters, words, watermarks in the prompt"""


def generate_prompt(name_zh, name_en, invention_zh, invention_en, description, scenario, language):
    user_message = f"""Child's invention card:
- Inventor: {name_zh} / {name_en}
- Invention (Chinese): {invention_zh}
- Invention (English): {invention_en}
- How it works: {description}
- Usage scenario: {scenario}

Generate the Imagen 4 prompt for this invention."""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=600
    )

    content = response.choices[0].message.content.strip()
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"prompt": content, "style_tags": []}


# ── Imagen 4 generation ────────────────────────────────────────────────────────

def generate_image_with_gemini(prompt_text):
    """Call Imagen 4 Fast via Gemini API, return raw image bytes (16:9)"""
    response = gemini_client.models.generate_images(
        model='imagen-4.0-fast-generate-001',
        prompt=prompt_text,
        config=google_types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio='16:9',
        )
    )
    return response.generated_images[0].image.image_bytes


# ── Poster composition ─────────────────────────────────────────────────────────

def compose_poster(bg_image_data, name_zh, name_en, invention_zh, invention_en, language):
    lang = TEXT_LAYERS.get(language, TEXT_LAYERS['zh'])
    W, H = 1920, 1080

    # Background
    if bg_image_data:
        bg = Image.open(io.BytesIO(bg_image_data)).convert('RGBA')
        bg = bg.resize((W, H), Image.LANCZOS)
    else:
        bg = Image.new('RGBA', (W, H), (10, 14, 30, 255))

    canvas = bg.copy()

    # Dark gradient overlays for text readability
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    # Bottom gradient (stronger)
    for i in range(500):
        alpha = int(200 * (i / 500) ** 1.5)
        ov_draw.rectangle([(0, H - 500 + i), (W, H - 500 + i + 1)], fill=(0, 0, 0, alpha))
    # Top-left vignette
    for i in range(500):
        alpha = int(130 * (i / 500))
        ov_draw.rectangle([(0, 0), (i, H)], fill=(0, 0, 0, min(alpha, 100)))
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    # ── Logo (top-left) ──
    pad = 52
    try:
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo_h = 90
        logo_w = int(logo_h * logo.width / logo.height)
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
        canvas.paste(logo, (pad, pad), logo)
        text_top = pad + logo_h + 18
    except Exception as e:
        print(f"Logo error: {e}")
        text_top = pad

    # ── Tagline PNG layer ──
    try:
        tagline_img = Image.open(lang['tagline']).convert('RGBA')
        canvas.paste(tagline_img, (pad, text_top), tagline_img)
        text_top += tagline_img.height + 8
    except Exception as e:
        print(f"Tagline error: {e}")

    # Credit is now merged into tagline PNG (three lines: tagline + CJK credit + EN credit)

    # ── Main text: bilingual name + invention (bottom-right) ──
    # Always show both CJK and English if both are provided
    font_cjk_inv  = None
    font_cjk_name = None
    font_en_inv   = None
    font_en_name  = None
    try:
        font_cjk_inv  = ImageFont.truetype(FONT_CJK_BLACK, 140)
        font_cjk_name = ImageFont.truetype(FONT_CJK_BLACK, 88)
        font_en_inv   = ImageFont.truetype(FONT_EN_BOLD,   110)
        font_en_name  = ImageFont.truetype(FONT_EN_BOLD,   68)
    except Exception as e:
        print(f"Font error: {e}")
        font_cjk_inv = font_cjk_name = font_en_inv = font_en_name = ImageFont.load_default()

    margin_r = 72
    margin_b = 56
    sh = 4

    # Determine what to show
    has_zh_inv  = bool(invention_zh and invention_zh.strip())
    has_en_inv  = bool(invention_en and invention_en.strip())
    has_zh_name = bool(name_zh and name_zh.strip())
    has_en_name = bool(name_en and name_en.strip())

    # --- Invention lines (large, bottom) ---
    # Primary: CJK if available, else English
    # Secondary: English sub-line if both exist
    if has_zh_inv:
        inv_primary      = invention_zh
        inv_primary_font = font_cjk_inv
        inv_sub          = (invention_en.upper() if has_en_inv else '')
        inv_sub_font     = font_en_inv
    else:
        inv_primary      = (invention_en.upper() if has_en_inv else '')
        inv_primary_font = font_en_inv
        inv_sub          = ''
        inv_sub_font     = None

    # --- Name lines (medium, above invention) ---
    if has_zh_name:
        name_primary      = name_zh
        name_primary_font = font_cjk_name
        name_sub          = (name_en if has_en_name else '')
        name_sub_font     = font_en_name
    else:
        name_primary      = (name_en if has_en_name else '')
        name_primary_font = font_en_name
        name_sub          = ''
        name_sub_font     = None

    def text_w(txt, fnt):
        if not txt or not fnt: return 0
        bb = draw.textbbox((0,0), txt, font=fnt)
        return bb[2] - bb[0]
    def text_h(txt, fnt):
        if not txt or not fnt: return 0
        bb = draw.textbbox((0,0), txt, font=fnt)
        return bb[3] - bb[1]

    # Calculate heights
    inv_p_h  = text_h(inv_primary, inv_primary_font) if inv_primary else 0
    inv_s_h  = text_h(inv_sub, inv_sub_font) if inv_sub else 0
    name_p_h = text_h(name_primary, name_primary_font) if name_primary else 0
    name_s_h = text_h(name_sub, name_sub_font) if name_sub else 0
    gap_inv  = 10 if inv_sub else 0
    gap_name = 6  if name_sub else 0
    gap_between = 22  # between name block and invention block

    total_h = name_p_h + name_s_h + gap_name + gap_between + inv_p_h + inv_s_h + gap_inv
    block_y = H - total_h - margin_b

    # Draw name primary
    cur_y = block_y
    if name_primary:
        nx = W - text_w(name_primary, name_primary_font) - margin_r
        draw.text((nx+sh, cur_y+sh), name_primary, font=name_primary_font, fill=(0,0,0,130))
        draw.text((nx,    cur_y),    name_primary, font=name_primary_font, fill=(255,255,255,210))
        cur_y += name_p_h + gap_name
    # Draw name sub
    if name_sub:
        nx = W - text_w(name_sub, name_sub_font) - margin_r
        draw.text((nx+sh, cur_y+sh), name_sub, font=name_sub_font, fill=(0,0,0,110))
        draw.text((nx,    cur_y),    name_sub, font=name_sub_font, fill=(200,220,240,180))
        cur_y += name_s_h

    cur_y += gap_between

    # Draw invention primary
    if inv_primary:
        ix = W - text_w(inv_primary, inv_primary_font) - margin_r
        draw.text((ix+sh, cur_y+sh), inv_primary, font=inv_primary_font, fill=(0,0,0,150))
        draw.text((ix,    cur_y),    inv_primary, font=inv_primary_font, fill=(255,255,255,255))
        cur_y += inv_p_h + gap_inv
    # Draw invention sub
    if inv_sub:
        ix = W - text_w(inv_sub, inv_sub_font) - margin_r
        draw.text((ix+sh, cur_y+sh), inv_sub, font=inv_sub_font, fill=(0,0,0,120))
        draw.text((ix,    cur_y),    inv_sub, font=inv_sub_font, fill=(180,210,240,200))

    # Output as PNG bytes
    final = canvas.convert('RGB')
    output = io.BytesIO()
    final.save(output, format='PNG', optimize=False)
    output.seek(0)
    return output.getvalue()


def dark_placeholder(W=1920, H=1080):
    bg = Image.new('RGB', (W, H), (8, 12, 28))
    draw = ImageDraw.Draw(bg)
    for i in range(H):
        r = int(8  + 18 * (i / H))
        g = int(12 + 12 * (i / H))
        b = int(28 + 35 * (i / H))
        draw.line([(0, i), (W, i)], fill=(r, g, b))
    buf = io.BytesIO()
    bg.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/bg_hero.jpg')
def serve_bg():
    return send_from_directory('static', 'bg_hero.jpg')

@app.route('/logo_light.png')
def serve_logo_light():
    return send_from_directory(ASSETS_DIR, 'logo_light.png')

@app.route('/logo_dark.png')
def serve_logo_dark():
    return send_from_directory(ASSETS_DIR, 'logo_dark.png')


@app.route('/api/generate-prompt', methods=['POST'])
def api_generate_prompt():
    data = request.json
    try:
        result = generate_prompt(
            data.get('name_zh', ''),
            data.get('name_en', ''),
            data.get('invention_zh', ''),
            data.get('invention_en', ''),
            data.get('description', ''),
            data.get('scenario', ''),
            data.get('language', 'zh')
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/compose-poster', methods=['POST'])
def api_compose_poster():
    try:
        name_zh      = request.form.get('name_zh', '')
        name_en      = request.form.get('name_en', '')
        invention_zh = request.form.get('invention_zh', '')
        invention_en = request.form.get('invention_en', '')
        language     = request.form.get('language', 'zh')

        bg_data = None
        if 'bg_image' in request.files:
            bg_data = request.files['bg_image'].read()

        poster_bytes = compose_poster(
            bg_data or dark_placeholder(),
            name_zh, name_en, invention_zh, invention_en, language
        )
        return jsonify({'success': True, 'poster': base64.b64encode(poster_bytes).decode()})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/one-shot', methods=['POST'])
def api_one_shot():
    """Full pipeline: card info → Prompt → Imagen 4 → Poster (one click)"""
    try:
        name_zh      = request.form.get('name_zh', '')
        name_en      = request.form.get('name_en', '')
        invention_zh = request.form.get('invention_zh', '')
        invention_en = request.form.get('invention_en', '')
        description  = request.form.get('description', '')
        scenario     = request.form.get('scenario', '')
        language     = request.form.get('language', 'zh')

        # Step 1: Generate prompt
        prompt_result = generate_prompt(
            name_zh, name_en, invention_zh, invention_en, description, scenario, language
        )
        prompt_text = prompt_result.get('prompt', '')

        # Step 2: Generate image with Imagen 4
        img_bytes = generate_image_with_gemini(prompt_text)

        # Step 3: Compose poster
        poster_bytes = compose_poster(
            img_bytes, name_zh, name_en, invention_zh, invention_en, language
        )

        return jsonify({
            'success': True,
            'prompt': prompt_text,
            'style_tags': prompt_result.get('style_tags', []),
            'poster': base64.b64encode(poster_bytes).decode(),
            'bg_image': base64.b64encode(img_bytes).decode(),
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/save-record', methods=['POST'])
def api_save_record():
    """Persist a text record (no images) to disk for admin review."""
    try:
        data = request.json or {}
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        rec  = {
            'time':         ts,
            'name_zh':      data.get('name_zh', ''),
            'name_en':      data.get('name_en', ''),
            'invention_zh': data.get('invention_zh', ''),
            'invention_en': data.get('invention_en', ''),
            'prompt':       data.get('prompt', ''),
            'style_tags':   data.get('style_tags', []),
        }
        fname = os.path.join(RECORDS_DIR, f'record_{ts}.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/admin/records')
def admin_records():
    """Simple admin page to view all collected records."""
    files = sorted([
        f for f in os.listdir(RECORDS_DIR) if f.endswith('.json')
    ], reverse=True)
    records = []
    for fn in files:
        try:
            with open(os.path.join(RECORDS_DIR, fn), encoding='utf-8') as f:
                records.append(json.load(f))
        except Exception:
            pass
    rows = ''.join([
        f"""<tr>
          <td>{r.get('time','')}</td>
          <td>{r.get('name_zh','')} / {r.get('name_en','')}</td>
          <td>{r.get('invention_zh','')} / {r.get('invention_en','')}</td>
          <td style='max-width:320px;font-size:11px;color:#888;word-break:break-all'>{r.get('prompt','')[:200]}…</td>
          <td>{', '.join(r.get('style_tags',[]))}</td>
        </tr>"""
        for r in records
    ])
    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
    <title>Sci-Dreamer Records</title>
    <style>
      body{{background:#0a0e1a;color:#cdd;font-family:monospace;padding:24px}}
      h1{{color:#00d4ff;margin-bottom:16px}}
      table{{border-collapse:collapse;width:100%}}
      th{{background:#111a2e;color:#00d4ff;padding:8px 12px;text-align:left;font-size:12px}}
      td{{padding:8px 12px;border-bottom:1px solid #1a2540;font-size:13px;vertical-align:top}}
      tr:hover td{{background:#0d1528}}
      .count{{color:#9b59ff;font-size:14px;margin-bottom:12px}}
    </style></head><body>
    <h1>Sci-Dreamer · 收集记录</h1>
    <div class='count'>共 {len(records)} 条记录</div>
    <table><thead><tr>
      <th>时间</th><th>姓名</th><th>发明</th><th>Prompt（前200字）</th><th>风格标签</th>
    </tr></thead><tbody>{rows}</tbody></table>
    </body></html>"""
    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
