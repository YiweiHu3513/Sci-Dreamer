# -*- coding: utf-8 -*-
import os
import io
import base64
import json
import re
import datetime
import hashlib
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, Response
from PIL import Image, ImageDraw, ImageFont
import requests
import time
from openai import OpenAI
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

app = Flask(__name__, static_folder='static')
app.secret_key = os.environ.get('ADMIN_SECRET_KEY', 'sci-dreamer-secret-2025')

# Admin password (set via env var ADMIN_PASSWORD, default: scidreamer2025)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'scidreamer2025')

def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect('/admin/login')
        return f(*args, **kwargs)
    return decorated

# DeepSeek client for prompt generation (text only, cheap)
deepseek_client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com',
)

# Vertex AI Imagen 4 — Service Account auth
_SA_JSON = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON', '')
_VERTEX_PROJECT = 'sci-dreamer-imagen'
_VERTEX_LOCATION = 'us-central1'
_VERTEX_MODEL = 'imagen-4.0-fast-generate-001'

def _get_vertex_token():
    """Get a short-lived OAuth2 access token from the service account JSON (base64 encoded)."""
    if not _SA_JSON:
        raise RuntimeError('GOOGLE_SERVICE_ACCOUNT_JSON env var is not set')
    # Support both raw JSON and base64-encoded JSON
    raw = _SA_JSON.strip()
    try:
        sa_info = json.loads(raw)
    except json.JSONDecodeError:
        # Try base64 decode
        sa_info = json.loads(base64.b64decode(raw).decode('utf-8'))
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    creds.refresh(GoogleAuthRequest())
    return creds.token

ASSETS_DIR   = os.path.join(os.path.dirname(__file__), 'assets')
RECORDS_DIR  = os.path.join(os.path.dirname(__file__), 'records')
POSTERS_DIR  = os.path.join(os.path.dirname(__file__), 'records', 'posters')
os.makedirs(RECORDS_DIR, exist_ok=True)
os.makedirs(POSTERS_DIR, exist_ok=True)

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

SYSTEM_PROMPT = """Convert a child's invention into an Imagen 4 image prompt. Output ONLY valid JSON:
{"prompt": "English image prompt under 150 words. Start with main subject. Include: photorealistic, 8K, volumetric lighting, sci-fi industrial aesthetic, speculative design. End with: 16:9, no text, no people, cinematic", "style_tags": ["sci-fi"]}
Rules: English only. Translate childlike metaphors to physical materials. No text/watermarks in prompt."""


def generate_prompt(name_zh, name_en, invention_zh, invention_en, description, scenario, language):
    user_message = f"""Child's invention card:
- Inventor: {name_zh} / {name_en}
- Invention (Chinese): {invention_zh}
- Invention (English): {invention_en}
- How it works: {description}
- Usage scenario: {scenario}

Generate the Imagen 4 prompt for this invention."""

    response = deepseek_client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ],
        temperature=0.7,
        max_tokens=400,
    )

    content = response.choices[0].message.content.strip()
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"prompt": content, "style_tags": []}


# ── Imagen 4 generation ────────────────────────────────────────────────────────

def generate_image_with_vertex(prompt_text):
    """Call Imagen 4 via Vertex AI REST API, return raw image bytes (16:9)"""
    token = _get_vertex_token()
    url = (
        f'https://{_VERTEX_LOCATION}-aiplatform.googleapis.com/v1'
        f'/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}'
        f'/publishers/google/models/{_VERTEX_MODEL}:predict'
    )
    payload = {
        'instances': [{'prompt': prompt_text}],
        'parameters': {
            'sampleCount': 1,
            'aspectRatio': '16:9',
            'addWatermark': False,
            'enhancePrompt': False,
        },
    }
    resp = requests.post(
        url,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    b64 = resp.json()['predictions'][0]['bytesBase64Encoded']
    return base64.b64decode(b64)


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
        img_bytes = generate_image_with_vertex(prompt_text)

        # Step 3: Compose poster
        poster_bytes = compose_poster(
            img_bytes, name_zh, name_en, invention_zh, invention_en, language
        )

        # Save record + poster to disk
        try:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]
            poster_filename = f'poster_{ts}.png'
            poster_path = os.path.join(POSTERS_DIR, poster_filename)
            with open(poster_path, 'wb') as pf:
                pf.write(poster_bytes)
            rec = {
                'time':         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'name_zh':      name_zh,
                'name_en':      name_en,
                'invention_zh': invention_zh,
                'invention_en': invention_en,
                'description':  description,
                'scenario':     scenario,
                'language':     language,
                'prompt':       prompt_text,
                'style_tags':   prompt_result.get('style_tags', []),
                'poster_file':  poster_filename,
            }
            rec_path = os.path.join(RECORDS_DIR, f'record_{ts}.json')
            with open(rec_path, 'w', encoding='utf-8') as rf:
                json.dump(rec, rf, ensure_ascii=False, indent=2)
        except Exception as save_err:
            print(f'Record save error: {save_err}')

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


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = ''
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        if pwd == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect('/admin/records')
        else:
            error = '密码错误，请重试'
    return f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Sci-Dreamer 后台登录</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#080c1a;display:flex;align-items:center;justify-content:center;min-height:100vh;font-family:"PingFang SC",sans-serif}}
  .card{{background:#0d1528;border:1px solid #1a2a4a;border-radius:16px;padding:48px 40px;width:360px;text-align:center}}
  h1{{color:#00d4ff;font-size:22px;margin-bottom:8px;letter-spacing:1px}}
  p{{color:#556;font-size:13px;margin-bottom:32px}}
  input{{width:100%;padding:12px 16px;background:#060a14;border:1px solid #1e3050;border-radius:8px;color:#cde;font-size:15px;outline:none;margin-bottom:16px}}
  input:focus{{border-color:#00d4ff}}
  button{{width:100%;padding:12px;background:linear-gradient(135deg,#00d4ff,#9b59ff);border:none;border-radius:8px;color:#fff;font-size:15px;font-weight:600;cursor:pointer;letter-spacing:1px}}
  .err{{color:#ff6b6b;font-size:13px;margin-top:12px}}
</style></head><body>
<div class="card">
  <h1>Sci-Dreamer</h1>
  <p>后台管理 · 请输入密码</p>
  <form method="POST">
    <input type="password" name="password" placeholder="请输入管理密码" autofocus>
    <button type="submit">进入后台</button>
  </form>
  <div class="err">{error}</div>
</div>
</body></html>'''


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect('/admin/login')


@app.route('/admin/poster/<filename>')
@require_admin
def serve_poster(filename):
    return send_from_directory(POSTERS_DIR, filename)


@app.route('/admin/records')
@require_admin
def admin_records():
    files = sorted([f for f in os.listdir(RECORDS_DIR) if f.endswith('.json')], reverse=True)
    records = []
    for fn in files:
        try:
            with open(os.path.join(RECORDS_DIR, fn), encoding='utf-8') as f:
                records.append(json.load(f))
        except Exception:
            pass

    rows = ''
    for r in records:
        poster_file = r.get('poster_file', '')
        if poster_file:
            poster_html = f'''<a href="/admin/poster/{poster_file}" target="_blank">
              <img src="/admin/poster/{poster_file}" style="width:200px;height:113px;object-fit:cover;border-radius:6px;border:1px solid #1a2a4a;display:block">
            </a>
            <a href="/admin/poster/{poster_file}" download style="color:#00d4ff;font-size:11px;margin-top:4px;display:block">下载 PNG</a>'''
        else:
            poster_html = '<span style="color:#444">无图</span>'

        rows += f'''<tr>
          <td style="white-space:nowrap">{r.get("time","")}</td>
          <td><b style="color:#fff">{r.get("name_zh","")}</b><br><span style="color:#778">{r.get("name_en","")}</span></td>
          <td><b style="color:#fff">{r.get("invention_zh","")}</b><br><span style="color:#778">{r.get("invention_en","")}</span></td>
          <td style="color:#556;font-size:12px">{r.get("description","")[:80]}</td>
          <td style="color:#556;font-size:12px">{r.get("scenario","")[:60]}</td>
          <td>{poster_html}</td>
        </tr>'''

    html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Sci-Dreamer 后台</title>
<style>
  *{{box-sizing:border-box}}
  body{{background:#080c1a;color:#cde;font-family:"PingFang SC",monospace;padding:32px;margin:0}}
  h1{{color:#00d4ff;font-size:24px;margin-bottom:4px}}
  .meta{{color:#445;font-size:13px;margin-bottom:24px}}
  .logout{{float:right;color:#ff6b6b;font-size:13px;text-decoration:none;padding:6px 14px;border:1px solid #ff6b6b;border-radius:6px}}
  table{{border-collapse:collapse;width:100%;margin-top:8px}}
  th{{background:#0d1528;color:#00d4ff;padding:10px 14px;text-align:left;font-size:12px;border-bottom:1px solid #1a2a4a;white-space:nowrap}}
  td{{padding:12px 14px;border-bottom:1px solid #0d1528;vertical-align:top;font-size:13px}}
  tr:hover td{{background:#0a1020}}
  .badge{{display:inline-block;background:#1a2a4a;color:#9b59ff;font-size:11px;padding:2px 8px;border-radius:4px;margin-bottom:4px}}
</style></head><body>
<a href="/admin/logout" class="logout">退出登录</a>
<h1>Sci-Dreamer · 发明卡收集后台</h1>
<div class="meta">共 <b style="color:#9b59ff">{len(records)}</b> 条记录 &nbsp;·&nbsp; 最新在前</div>
<table>
  <thead><tr>
    <th>生成时间</th>
    <th>发明者</th>
    <th>发明名称</th>
    <th>原理描述</th>
    <th>使用场景</th>
    <th>海报</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>'''
    return html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
