# pixel_art_compare_fixed.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import csv
import webcolors
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm as mm_unit
import base64
import pandas as pd

# Optional slider package
try:
    from streamlit_image_comparison import image_comparison
except Exception:
    image_comparison = None

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Pixel Art Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎨"
)

# ------------------ CSS FIXED FOR VISIBILITY ------------------
st.markdown("""
<style>
    /* 1. Force the main background to your Graph Paper look */
    .stApp {
        background-color: #fdfbf7;
        background-image: linear-gradient(#e5e7eb 1px, transparent 1px), linear-gradient(90deg, #e5e7eb 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* 2. FORCE DARK TEXT EVERYWHERE */
    /* This overrides Streamlit's Dark Mode white text settings */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp span, .stApp div, .stApp label, .stMarkdown, .stText {
        color: #333333 !important;
    }

    /* 3. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        box-shadow: 2px 0 5px rgba(0,0,0,0.02);
    }
    [data-testid="stSidebar"] * {
        color: #333333 !important;
    }

    /* 4. Fix Input Widgets (Number Input, Text Input, Selectbox) */
    /* These often turn white in dark mode, so we force them back to dark */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-color: #d1d5db !important;
    }
    input[type="number"], input[type="text"] {
        color: #333333 !important;
        -webkit-text-fill-color: #333333 !important; 
        background-color: transparent !important;
    }
    div[data-baseweb="select"] span {
        color: #333333 !important;
    }
    
    /* 5. Fix File Uploader Text */
    [data-testid="stFileUploader"] section {
        background-color: #f8fafc !important;
        border: 1px dashed #cbd5e1 !important;
    }
    [data-testid="stFileUploader"] small {
        color: #64748b !important;
    }

    /* 6. Hero Image Styling */
    .hero-wrap { display:flex; justify-content:center; margin: 18px 0 8px 0; }
    .hero-wrap img {
        max-width: 420px !important;
        width: auto !important;
        height: auto !important;
        border-radius: 14px !important;
        border: 6px solid white !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
    }

    /* 7. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        height:45px; 
        background:#fff; 
        border-radius:8px 8px 0 0; 
        border:1px solid #e5e7eb; 
        color:#64748b !important; 
        font-weight:600; 
        padding:0 20px; 
    }
    .stTabs [aria-selected="true"] { 
        color: #4f46e5 !important; 
        border-top: 3px solid #4f46e5 !important; 
    }
    
    /* Ensure tab content remains visible */
    .stTabs [data-baseweb="tab-panel"] {
        color: #333333 !important;
    }

    /* 8. Misc Overrides */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    @media (max-width: 600px) { .hero-wrap img { max-width: 92% !important; border-radius:10px !important; border: 4px solid white !important; } }
    [data-testid="stImage"] img { display: block; z-index: 2; position: relative; }
    iframe { z-index: 2; position: relative; background: transparent; }

</style>
""", unsafe_allow_html=True)

# ------------------ Utilities ------------------
def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def closest_css3_name(rgb):
    r1, g1, b1 = [int(x) for x in rgb]
    try:
        name = webcolors.rgb_to_name((r1, g1, b1), spec='css3')
        return name, 0.0
    except Exception:
        pass
    candidates = []
    if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
        hexmap = webcolors.CSS3_HEX_TO_NAMES
        for hexcode, name in hexmap.items():
            try:
                r2, g2, b2 = webcolors.hex_to_rgb(hexcode)
                candidates.append((name, (r2, g2, b2)))
            except: continue
    else:
        fallback = {"#000000":"black","#ffffff":"white","#ff0000":"red","#00ff00":"lime","#0000ff":"blue"}
        for hexc, name in fallback.items():
            try:
                r2, g2, b2 = webcolors.hex_to_rgb(hexc)
                candidates.append((name, (r2, g2, b2)))
            except: continue
    min_dist = float("inf"); best = "unknown"
    for name, (r2, g2, b2) in candidates:
        d = (r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2
        if d < min_dist:
            min_dist = d; best = name
    return best, (min_dist ** 0.5 if min_dist is not None else None)

def pil_image_to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO(); img.save(buf, format=fmt); buf.seek(0); return buf

def get_text_size(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont):
    try:
        bbox = draw.textbbox((0,0), text, font=font); return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        return (len(text)*6, 10)

# ------------------ Quantization & Rendering ------------------
def quantize_pil_mediancut(img_rgb, target_w, target_h, n_colors):
    small = img_rgb.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    quant = small.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    arr = np.array(quant)
    palette_flat = quant.getpalette() or []
    unique = np.unique(arr); pal_list = []
    for idx in unique:
        idx = int(idx)
        if idx*3+2 < len(palette_flat):
            r = palette_flat[idx*3]; g = palette_flat[idx*3+1]; b = palette_flat[idx*3+2]
        else:
            r,g,b = (0,0,0)
        pal_list.append((idx, (int(r), int(g), int(b))))
    counts = [(int((arr==u).sum()), u) for u in unique]; counts.sort(reverse=True)
    ordered = []
    for cnt, u in counts:
        for pidx, rgb in pal_list:
            if pidx == int(u):
                ordered.append((int(u), rgb, int(cnt)))
                break
    return arr, ordered

def quantize_kmeans(img_rgb, target_w, target_h, n_colors, random_state=0):
    small = img_rgb.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    arr_rgb = np.array(small).reshape(-1,3).astype(float)
    k = min(n_colors, len(arr_rgb)); 
    if k <= 0: k = 1
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(arr_rgb)
    centers = np.clip(kmeans.cluster_centers_.round().astype(int), 0, 255)
    label_grid = labels.reshape((target_h, target_w))
    unique, counts = np.unique(labels, return_counts=True); order = np.argsort(-counts)
    pal_list = []
    for rank_idx in order:
        idx = int(unique[rank_idx]); rgb = tuple(int(x) for x in centers[idx]); cnt = int(counts[rank_idx])
        pal_list.append((int(idx), rgb, cnt))
    return label_grid, pal_list, centers

def render_block_image(labels, palette_map, scale=16, draw_grid=False):
    h, w = labels.shape; out = Image.new("RGB", (w*scale, h*scale), "white"); draw = ImageDraw.Draw(out)
    for y in range(h):
        for x in range(w):
            lab = int(labels[y, x]); color = tuple(palette_map.get(lab, (0,0,0)))
            x0, y0 = x*scale, y*scale
            draw.rectangle([x0, y0, x0+scale, y0+scale], fill=color)
    if draw_grid:
        linew = max(1, scale//12); grid_color = (255,255,255)
        for i in range(1, w): draw.line([(i*scale, 0),(i*scale, h*scale)], fill=grid_color, width=linew)
        for j in range(1, h): draw.line([(0, j*scale),(w*scale, j*scale)], fill=grid_color, width=linew)
    return out

def render_block_image_with_numbers(labels, palette_map, rank_map, scale=16):
    h, w = labels.shape; out = Image.new("RGB", (w*scale, h*scale), "white"); draw = ImageDraw.Draw(out)
    try: fnt = ImageFont.truetype("DejaVuSans.ttf", max(10, scale//2))
    except: fnt = ImageFont.load_default()
    for y in range(h):
        for x in range(w):
            lab = int(labels[y, x]); color = tuple(palette_map.get(lab, (0,0,0)))
            x0, y0 = x*scale, y*scale
            draw.rectangle([x0, y0, x0+scale, y0+scale], fill=color)
    linew = max(1, scale//12); grid_color = (255,255,255)
    for i in range(1, w): draw.line([(i*scale,0),(i*scale,h*scale)], fill=grid_color, width=linew)
    for j in range(1, h): draw.line([(0,j*scale),(w*scale,j*scale)], fill=grid_color, width=linew)
    for y in range(h):
        for x in range(w):
            lab = int(labels[y, x]); rank = rank_map.get(lab, "?"); txt = str(rank)
            cx = int((x+0.5)*scale); cy = int((y+0.5)*scale)
            wtxt, htxt = get_text_size(draw, txt, fnt); pos = (cx - wtxt//2, cy - htxt//2)
            block_rgb = palette_map.get(lab, (255,255,255))
            lum = 0.299*block_rgb[0] + 0.587*block_rgb[1] + 0.114*block_rgb[2]
            txt_color = (0,0,0) if lum > 160 else (255,255,255)
            outline = (255,255,255) if txt_color == (0,0,0) else (0,0,0)
            for ox in (-1,0,1):
                for oy in (-1,0,1):
                    draw.text((pos[0]+ox,pos[1]+oy), txt, font=fnt, fill=outline)
            draw.text(pos, txt, font=fnt, fill=txt_color)
    return out

def palette_swatch_image(palette_ordered, palette_map, sw=96, sh=96, cols=6):
    n = len(palette_ordered); cols = min(cols, n) if n>0 else 1; rows = (n + cols - 1) // cols
    w = cols * sw; h = rows * sh + 36; im = Image.new("RGB", (w,h), "white"); draw = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except: font = ImageFont.load_default()
    for idx, (lab, rgb, cnt) in enumerate(palette_ordered):
        row = idx // cols; col = idx % cols; x = col * sw; y = row * sh
        fill_rgb = tuple(palette_map.get(int(lab), rgb))
        draw.rectangle([x+8, y+8, x+sw-8, y+sh-8], fill=fill_rgb)
        hexc = rgb_to_hex(fill_rgb); name, _ = closest_css3_name(fill_rgb)
        text = f"{idx+1}. {name}\n{hexc}\n{cnt} tiles"
        lum = 0.299*fill_rgb[0] + 0.587*fill_rgb[1] + 0.114*fill_rgb[2]
        text_color = (0,0,0) if lum > 160 else (255,255,255)
        draw.multiline_text((x+12, y+sh-48), text, fill=text_color, font=font)
    draw.text((4,h-28), f"Palette ({n} colors)", fill=(0,0,0), font=font); return im

def create_pattern_pdf(labels, palette_ordered, palette_map, tile_mm, page_orientation='portrait', page_size=A4):
    h, w = labels.shape; pts_per_mm = 72.0 / 25.4; tile_pts = tile_mm * pts_per_mm
    page = landscape(page_size) if page_orientation=='landscape' else portrait(page_size)
    page_w, page_h = page; margin = 12 * mm_unit; usable_w = page_w - 2*margin; usable_h = page_h - 2*margin
    grid_w_pts = w * tile_pts; grid_h_pts = h * tile_pts; scale = 1.0
    if grid_w_pts > usable_w or grid_h_pts > usable_h * 0.7:
        scale_x = usable_w / grid_w_pts; scale_y = (usable_h * 0.7) / grid_h_pts; scale = min(scale_x, scale_y)
        tile_pts *= scale; grid_w_pts = w * tile_pts; grid_h_pts = h * tile_pts
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=page)
    c.setTitle("Mosaic Pattern"); c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, page_h - margin + 2*mm_unit, "Mosaic Pattern Guide"); c.setFont("Helvetica", 9)
    total_w_cm = (w * tile_mm) / 10.0; total_h_cm = (h * tile_mm) / 10.0
    c.drawString(margin, page_h - margin - 8, f"Dimensions: {w}x{h} tiles • Physical Size: {total_w_cm:.1f} x {total_h_cm:.1f} cm")
    grid_x = margin + (usable_w - grid_w_pts)/2.0; grid_y = page_h - margin - 40 - grid_h_pts
    for row in range(h):
        for col in range(w):
            lab = int(labels[row, col]); rgb = palette_map.get(lab, (0,0,0)); r,g,b = [v/255.0 for v in rgb]
            x = grid_x + col * tile_pts; y = grid_y + (h - 1 - row) * tile_pts
            c.setFillColorRGB(r,g,b); c.rect(x, y, tile_pts, tile_pts, stroke=0, fill=1)
            c.setStrokeColorRGB(1,1,1); c.setLineWidth(max(0.2, tile_pts*0.02)); c.rect(x, y, tile_pts, tile_pts, stroke=1, fill=0)
            c.setFont("Helvetica", max(6, int(tile_pts*0.35)))
            rank = 0
            for rank_idx, (label_id, _, _) in enumerate(palette_ordered, start=1):
                if label_id == lab: rank = rank_idx; break
            text = str(rank); tw = c.stringWidth(text, "Helvetica", max(6, int(tile_pts*0.35)))
            tx = x + (tile_pts - tw)/2.0; ty = y + (tile_pts*0.5) - (max(6, int(tile_pts*0.35))/3.0)
            lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            if lum > 160: c.setFillColorRGB(0,0,0)
            else: c.setFillColorRGB(1,1,1)
            c.drawString(tx, ty, text)
    # legend
    legend_x = margin; legend_y = grid_y - 20*mm_unit
    if legend_y < margin:
        legend_x = grid_x + grid_w_pts + 10*mm_unit; legend_y = page_h - margin - 20*mm_unit
    c.setFont("Helvetica-Bold", 10); c.setFillColorRGB(0,0,0); c.drawString(legend_x, legend_y, "Color Palette Legend")
    oy = legend_y - 12; sw = 10 * mm_unit; sh = 10 * mm_unit; gap_y = 4 * mm_unit
    for rank_idx, (_, rgbc, _) in enumerate(palette_ordered, start=1):
        x = legend_x; y = oy - (rank_idx-1)*(sh + gap_y)
        if y < margin: break
        r,g,b = [v/255.0 for v in rgbc]; c.setFillColorRGB(r,g,b); c.rect(x, y - sh, sw, sh, stroke=0, fill=1)
        lum = 0.299*rgbc[0] + 0.587*rgbc[1] + 0.114*rgbc[2]
        if lum > 160: text_col = (0,0,0)
        else: text_col = (1,1,1)
        c.setFillColorRGB(*text_col)
        c.setFont("Helvetica", 8)
        name, _ = closest_css3_name(rgbc)
        c.drawString(x + sw + 6, y - sh + sh/2 - 3, f"{rank_idx}. {name} {rgb_to_hex(rgbc)}")
    c.showPage(); c.save(); buf.seek(0); return buf

# ------------------ Session state ------------------
for key in ['labels','palette_ordered','palette_map','pixel_bytes','numbered_bytes','pal_bytes','pdf_bytes','csv_text','last_uploaded_name','original_preview']:
    if key not in st.session_state:
        st.session_state[key] = None

# ------------------ UI ------------------
st.title("🎨 Pixel Art Studio")

# Hero image
local_hero = "/mnt/data/a353a312-5cba-4b4f-b9c4-1cc41bc4f954.png"
try:
    Image.open(local_hero)
    hero_used = local_hero
except Exception:
    hero_used = "https://ik.imagekit.io/n14y9ertp/PA.png"
st.markdown(f'<div class="hero-wrap"><img src="{hero_used}" alt="hero"></div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; font-weight: 400; color: #475569 !important;'>Transform photos into pixel art, mosaic patterns, and printable guides.</p>", unsafe_allow_html=True)
st.divider()

col_controls, col_preview = st.columns([1, 2], gap="large")

with col_controls:
    st.subheader("Settings")
    uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded:
        if uploaded.name != st.session_state['last_uploaded_name']:
            for k in ['labels','pixel_bytes','numbered_bytes','pal_bytes','pdf_bytes','csv_text','original_preview','palette_ordered','palette_map']:
                st.session_state[k] = None
            st.session_state['last_uploaded_name'] = uploaded.name

    st.markdown("### Process")
    quant_method = st.radio("Color Method", ["K-Means","PIL (Median Cut)"], index=0, horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        preset = st.selectbox("Grid Size", ["16×16","32×32","48×48","64×64","Custom"], index=2)
    with c2:
        if preset == "Custom":
            target_px = st.number_input("Px size", 8, 512, 32)
        else:
            target_px = int(preset.split("×")[0])
    n_colors = st.slider("Color Count", 2, 64, 12)
    keep_aspect = st.checkbox("Maintain aspect ratio", value=True)

    with st.expander("📄 PDF & Export Options"):
        tile_mm = st.number_input("Tile Size (mm)", 1.0, 100.0, 10.0, 0.5)
        page_orient = st.radio("Orientation", ["portrait","landscape"], horizontal=True)
        display_scale = st.slider("Screen Preview Scale", 4, 32, 16)

    generate_btn = st.button("✨ Generate Pixel Art", type="primary", use_container_width=True)

# ------------------ Logic ------------------
if generate_btn and uploaded:
    with st.spinner("Processing image..."):
        try:
            img = Image.open(uploaded).convert("RGB")
            w0, h0 = img.size
            if keep_aspect:
                if w0 >= h0:
                    new_w = target_px
                    new_h = max(1, int(round(h0 * (target_px / w0))))
                else:
                    new_h = target_px
                    new_w = max(1, int(round(w0 * (target_px / h0))))
            else:
                new_w = new_h = target_px

            if quant_method == "PIL (Median Cut)":
                labels_arr, palette_ordered = quantize_pil_mediancut(img, new_w, new_h, n_colors)
            else:
                labels_arr, palette_ordered, centers = quantize_kmeans(img, new_w, new_h, n_colors)

            st.session_state['labels'] = labels_arr
            st.session_state['palette_ordered'] = palette_ordered
            palette_map = {int(lab): tuple(rgb) for lab, rgb, cnt in palette_ordered}
            st.session_state['palette_map'] = palette_map
            rank_map = {int(lab): idx for idx, (lab, rgb, cnt) in enumerate(palette_ordered, start=1)}

            img_pixel = render_block_image(labels_arr, palette_map, scale=display_scale, draw_grid=True)
            img_numbered = render_block_image_with_numbers(labels_arr, palette_map, rank_map, scale=display_scale)
            pal_im = palette_swatch_image(palette_ordered, palette_map, sw=100, sh=100, cols=4)

            st.session_state['pixel_bytes'] = pil_image_to_bytes(img_pixel).getvalue()
            st.session_state['numbered_bytes'] = pil_image_to_bytes(img_numbered).getvalue()
            st.session_state['pal_bytes'] = pil_image_to_bytes(pal_im).getvalue()

            try:
                orig_preview = img.resize(img_pixel.size, resample=Image.Resampling.LANCZOS)
            except Exception:
                orig_preview = img.copy().convert("RGB")
            st.session_state['original_preview'] = orig_preview

            total_blocks = labels_arr.size
            table_rows = []
            for i, (lab, rgb, cnt) in enumerate(palette_ordered):
                name = closest_css3_name(rgb)[0] or ""
                table_rows.append({"Rank": i+1, "Name": name, "Hex": rgb_to_hex(rgb), "Count": cnt, "Percent": f"{(cnt/total_blocks*100):.1f}%"})
            st.session_state['csv_text'] = None
            try:
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["Rank","Name","Hex","Count","Percent"])
                writer.writeheader(); writer.writerows(table_rows)
                st.session_state['csv_text'] = csv_buf.getvalue()
            except Exception:
                st.session_state['csv_text'] = None

            pdf_buf = create_pattern_pdf(labels_arr, palette_ordered, palette_map, tile_mm, page_orient)
            st.session_state['pdf_bytes'] = pdf_buf.getvalue()

            st.session_state['_palette_df'] = pd.DataFrame(table_rows) if table_rows else None

        except Exception as e:
            st.error(f"Error processing image: {e}")

# ------------------ Display ------------------
with col_preview:
    if st.session_state.get('pixel_bytes'):
        tabs = st.tabs(["🖼️ Comparison", "🖼️ Pixel Image", "🔢 Numbered Grid", "🎨 Palette", "📄 Printable PDF"])
        with tabs[0]:
            if st.session_state.get('original_preview') is not None and st.session_state.get('pixel_bytes') is not None:
                if image_comparison is not None:
                    try:
                        image_comparison(
                            img1=st.session_state['original_preview'],
                            img2=Image.open(io.BytesIO(st.session_state['pixel_bytes'])),
                            label1="Original Photo",
                            label2="Pixel Art Output",
                            starting_position=50,
                            show_labels=True,
                            make_responsive=True
                        )
                    except Exception as e:
                        st.error(f"Comparison widget error: {e}")
                        c1, c2 = st.columns(2)
                        with c1: st.image(st.session_state['original_preview'], caption="Original")
                        with c2: st.image(st.session_state['pixel_bytes'], caption="Pixel Art")
                else:
                    st.info("Install `streamlit-image-comparison` for the draggable slider. Showing side-by-side fallback.")
                    c1, c2 = st.columns(2)
                    with c1: st.image(st.session_state['original_preview'], caption="Original Photo")
                    with c2: st.image(st.session_state['pixel_bytes'], caption="Pixel Art Output")
            else:
                st.info("Comparison requires an uploaded image and a generated pixel art. Upload and generate first.")

        with tabs[1]:
            if st.session_state.get('pixel_bytes') is not None:
                # CHANGED: Replaced use_column_width=False with use_container_width=False
                st.image(st.session_state['pixel_bytes'], caption="Final Pixel Art", use_container_width=False)
                st.download_button("Download Pixel Art", st.session_state['pixel_bytes'], "pixel_art.png", "image/png")
            else:
                st.info("No pixel art generated yet.")

        with tabs[2]:
            if st.session_state.get('numbered_bytes') is not None:
                # CHANGED: If you want this to expand to container width, set use_container_width=True
                # Keeping default behavior similar to previous code
                st.image(st.session_state['numbered_bytes'], caption="Numbered Guide")
                st.download_button("Download Guide Image", st.session_state['numbered_bytes'], "pixel_guide.png", "image/png")
            else:
                st.info("No numbered guide yet.")

        with tabs[3]:
            if st.session_state.get('pal_bytes') is not None:
                # CHANGED: Replaced use_container_width=True (already correct in newer versions, but double check)
                st.image(st.session_state['pal_bytes'], use_container_width=True)
                if st.session_state.get('_palette_df') is not None:
                    st.dataframe(st.session_state['_palette_df'], use_container_width=True)
                if st.session_state.get('csv_text') is not None:
                    st.download_button("Download CSV", st.session_state['csv_text'], "palette.csv", "text/csv")
            else:
                st.info("No palette to show yet.")

        with tabs[4]:
            if st.session_state.get('pdf_bytes') is not None:
                b64 = base64.b64encode(st.session_state['pdf_bytes']).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.download_button("Download PDF Pattern", st.session_state['pdf_bytes'], "pattern.pdf", "application/pdf")
            else:
                st.info("No PDF generated yet.")
    else:
        st.info("Upload an image and click Generate to see results.")

