#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
animate_map.py — Real geographic map animation of the Santa Barbara water system.

Saves the Cartopy map as a static PNG once, then composites animated
overlays on top using PIL with smooth interpolation between keyframes.

Usage (from project root):
    python3 src/animate_map.py

Output:
    result/rl/animation/map_animation.gif
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'result', 'rl', 'trajectories')
ANIM_DIR = os.path.join(PROJECT_ROOT, 'result', 'rl', 'animation')
os.makedirs(ANIM_DIR, exist_ok=True)

# ── Load trajectories ─────────────────────────────────────────────────────────
print("Loading trajectories...")
cost_data = np.load(os.path.join(TRAJ_DIR, 'cost_only_drought.npz'))
safe_data = np.load(os.path.join(TRAJ_DIR, 'safe_drought.npz'))

c_sc    = cost_data['sc'];    s_sc    = safe_data['sc']
c_sgi   = cost_data['sgi'];   s_sgi   = safe_data['sgi']
c_sswp  = cost_data['sswp'];  s_sswp  = safe_data['sswp']
c_desal = cost_data['desal']; s_desal = safe_data['desal']
c_risk  = cost_data['risk'];  s_risk  = safe_data['risk']
c_cost  = cost_data['cost'];  s_cost  = safe_data['cost']

desal_cap  = float(cost_data['desal_capacity'])
SC_MAX, SGI_MAX, SSWP_MAX = 20000.0, 4550.0, 7500.0
SAFETY_THR = 3.0

H      = len(c_sc)
STEP   = 6    # real data keyframes every 6 months
INTERP = 4    # interpolated frames between each keyframe

frames = list(range(0, H, STEP))
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

# Build interpolated frame list: (t0, t1, alpha)
interp_frames = []
for i in range(len(frames)):
    t0 = frames[i]
    t1 = frames[i+1] if i+1 < len(frames) else frames[i]
    for k in range(INTERP):
        alpha = k / INTERP
        interp_frames.append((t0, t1, alpha))

print(f"  {H} months  |  {len(frames)} keyframes  |  "
      f"{len(interp_frames)} interpolated frames")

# ── Real GPS coordinates ──────────────────────────────────────────────────────
LON_CACHUMA = -119.9382;  LAT_CACHUMA = 34.5770
LON_GIBR    = -119.6615;  LAT_GIBR    = 34.5259
LON_CITY    = -119.6982;  LAT_CITY    = 34.4208
LON_DESAL   = -119.7100;  LAT_DESAL   = 34.3980
LON_SWP     = -119.9000;  LAT_SWP     = 34.6500

LON_MIN, LON_MAX = -120.15, -119.45
LAT_MIN, LAT_MAX =  34.30,   34.72
PROJ = ccrs.PlateCarree()

# ── Colour helpers ────────────────────────────────────────────────────────────
WATER_CMAP = LinearSegmentedColormap.from_list(
    'water', ['#8B0000','#FF6B35','#FFD700','#4FC3F7','#0277BD'])

def water_color(frac):
    rgba = WATER_CMAP(np.clip(frac, 0, 1))
    return tuple(int(c*255) for c in rgba[:3])

def risk_color_rgb(months):
    if months < SAFETY_THR:  return (244, 67,  54)
    elif months < 6:          return (255,152,   0)
    elif months < 12:         return (255,215,   0)
    else:                     return ( 76,175,  80)

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))


# ── Step 1: Render and cache static map background PNG ───────────────────────
BG_PATH = os.path.join(ANIM_DIR, '_map_background.png')

def render_background():
    print("Rendering map background (once)...")
    fig, axes = plt.subplots(
        1, 2, figsize=(20, 8.7),
        subplot_kw={'projection': PROJ},
        gridspec_kw={'wspace': 0.04}
    )
    fig.patch.set_facecolor('#0d1b2a')

    for ax in axes:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PROJ)
        try:
            tiler = cimgt.OSM()
            ax.add_image(tiler, 11)
        except Exception:
            ax.set_facecolor('#2d6a4f')
            ax.add_feature(cfeature.OCEAN.with_scale('10m'),
                           facecolor='#023e8a', zorder=1)
            ax.add_feature(cfeature.LAND.with_scale('10m'),
                           facecolor='#606c38', zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                           edgecolor='#e9c46a', linewidth=1, zorder=2)
            ax.add_feature(cfeature.RIVERS.with_scale('10m'),
                           edgecolor='#90e0ef', linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.LAKES.with_scale('10m'),
                           facecolor='#90e0ef', zorder=2)

        ax.add_patch(mpatches.Rectangle(
            (LON_MIN, LAT_MIN), LON_MAX-LON_MIN, LAT_MAX-LAT_MIN,
            transform=PROJ, facecolor='#000020', alpha=0.28, zorder=3))

        gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                          color='white', alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 6, 'color': 'white'}
        gl.ylabel_style = {'size': 6, 'color': 'white'}

    plt.tight_layout(pad=0.2)
    fig.savefig(BG_PATH, dpi=100, bbox_inches='tight',
                facecolor='#0d1b2a')
    plt.close(fig)
    print(f"  Background saved → {BG_PATH}")

if not os.path.exists(BG_PATH):
    render_background()
else:
    print(f"  Reusing cached background → {BG_PATH}")

bg_full  = Image.open(BG_PATH).convert('RGBA')
BG_W, BG_H_px = bg_full.size
PANEL_W  = BG_W // 2
PANEL_H  = BG_H_px
BAR_H    = 80
FRAME_W  = BG_W
FRAME_H  = PANEL_H + BAR_H

bg_left  = bg_full.crop((0,       0, PANEL_W, PANEL_H))
bg_right = bg_full.crop((PANEL_W, 0, BG_W,    PANEL_H))
print(f"  Panel size: {PANEL_W} x {PANEL_H} px")


# ── Coordinate helpers ────────────────────────────────────────────────────────
def geo_to_px(lon, lat, pw=PANEL_W, ph=PANEL_H):
    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * pw)
    y = int((1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * ph)
    return x, y


# ── Fonts ─────────────────────────────────────────────────────────────────────
try:
    FONT_SM  = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
    FONT_MED = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    FONT_LG  = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
    FONT_XL  = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 28)
except Exception:
    FONT_SM = FONT_MED = FONT_LG = FONT_XL = ImageFont.load_default()


# ── PIL drawing helpers ───────────────────────────────────────────────────────
def draw_reservoir(draw, cx, cy, radius, frac, label, vol):
    frac  = np.clip(frac, 0, 1)
    wc    = water_color(frac)
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                 fill=(38,50,56,200))
    if frac > 0.01:
        draw.pieslice([cx-radius, cy-radius, cx+radius, cy+radius],
                      start=-90, end=-90+360*frac,
                      fill=wc+(220,))
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                 outline=(207,216,220,255), width=2)
    pct = f'{frac*100:.0f}%'
    bb  = draw.textbbox((0,0), pct, font=FONT_SM)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx-tw//2, cy-th//2), pct,
              fill=(255,255,255,255), font=FONT_SM)
    for i, line in enumerate(f'{label}\n{vol:,.0f} AF'.split('\n')):
        bb = draw.textbbox((0,0), line, font=FONT_SM)
        lw = bb[2]-bb[0]
        draw.text((cx-lw//2, cy+radius+4+i*14), line,
                  fill=(255,255,255,210), font=FONT_SM)


def draw_pipe(draw, lon1, lat1, lon2, lat2, active, color_hex,
              pw=PANEL_W, ph=PANEL_H):
    x1,y1 = geo_to_px(lon1, lat1, pw, ph)
    x2,y2 = geo_to_px(lon2, lat2, pw, ph)
    r,g,b = hex_to_rgb(color_hex)
    alpha = 210 if active else 45
    width = 3   if active else 1
    draw.line([x1,y1,x2,y2], fill=(r,g,b,alpha), width=width)


def draw_desal(draw, lon, lat, des_f, pw=PANEL_W, ph=PANEL_H):
    x,y = geo_to_px(lon, lat, pw, ph)
    intensity = np.clip(des_f, 0, 1)
    color = (0,229,255,220) if intensity > 0.15 else (84,110,122,180)
    r = int(8 + 18*intensity)
    draw.polygon([(x,y-r),(x+r,y),(x,y+r),(x-r,y)],
                 fill=color, outline=(255,255,255,180))
    label = f'Desal\n{des_f*desal_cap:.0f} AF/mo'
    for i, line in enumerate(label.split('\n')):
        bb = draw.textbbox((0,0), line, font=FONT_SM)
        lw = bb[2]-bb[0]
        draw.text((x-lw//2, y+r+3+i*13), line,
                  fill=(0,229,255,220), font=FONT_SM)


def draw_city(draw, lon, lat, pw=PANEL_W, ph=PANEL_H):
    x,y = geo_to_px(lon, lat, pw, ph)
    r   = 10
    draw.polygon([(x,y-r),(x+r//2,y+r//2),(x-r//2,y+r//2)],
                 fill=(255,143,0,230), outline=(255,255,255,200))
    draw.text((x+13, y-7), 'Santa Barbara',
              fill=(255,255,255,230), font=FONT_SM)


def draw_badge(draw, x, y, text, color_rgb, font=None):
    font = font or FONT_MED
    r,g,b = color_rgb
    bb  = draw.textbbox((0,0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    pad = 6
    draw.rounded_rectangle(
        [x-pad, y-pad, x+tw+pad, y+th+pad],
        radius=6, fill=(13,27,42,210), outline=(r,g,b,220), width=2)
    draw.text((x,y), text, fill=(r,g,b,255), font=font)


def render_panel(bg_crop, sc_f, sgi_f, sswp_f, des_f,
                 risk_val, total_cost_m, title, accent_rgb):
    panel = bg_crop.copy().convert('RGBA')
    draw  = ImageDraw.Draw(panel, 'RGBA')
    pw, ph = panel.size

    # Pipes
    draw_pipe(draw, LON_CACHUMA, LAT_CACHUMA, LON_CITY, LAT_CITY,
              active=sc_f>0.05,   color_hex='#4FC3F7', pw=pw, ph=ph)
    draw_pipe(draw, LON_GIBR,    LAT_GIBR,    LON_CITY, LAT_CITY,
              active=sgi_f>0.05,  color_hex='#29B6F6', pw=pw, ph=ph)
    draw_pipe(draw, LON_SWP,     LAT_SWP,     LON_CITY, LAT_CITY,
              active=sswp_f>0.05, color_hex='#0288D1', pw=pw, ph=ph)
    draw_pipe(draw, LON_DESAL,   LAT_DESAL,   LON_CITY, LAT_CITY,
              active=des_f>0.15,  color_hex='#00E5FF', pw=pw, ph=ph)

    # Reservoirs
    cx,cy = geo_to_px(LON_CACHUMA, LAT_CACHUMA, pw, ph)
    draw_reservoir(draw, cx, cy, 55, sc_f,   'Cachuma',   sc_f*SC_MAX)
    cx,cy = geo_to_px(LON_GIBR, LAT_GIBR, pw, ph)
    draw_reservoir(draw, cx, cy, 38, sgi_f,  'Gibraltar', sgi_f*SGI_MAX)
    cx,cy = geo_to_px(LON_SWP, LAT_SWP, pw, ph)
    draw_reservoir(draw, cx, cy, 32, sswp_f, 'SWP',       sswp_f*SSWP_MAX)

    # City and desal
    draw_city(draw, LON_CITY,  LAT_CITY,  pw, ph)
    draw_desal(draw, LON_DESAL, LAT_DESAL, des_f, pw, ph)

    # Risk badge bottom left
    rc = risk_color_rgb(risk_val)
    draw_badge(draw, 12, ph-42, f'Risk: {risk_val:.1f} mo', rc)

    # Cost badge bottom right
    cost_text = f'${total_cost_m:,.0f}M'
    bb = draw.textbbox((0,0), cost_text, font=FONT_MED)
    tw = bb[2]-bb[0]
    draw_badge(draw, pw-tw-24, ph-42, cost_text, (255,215,0))

    # Title top centre
    r,g,b = accent_rgb
    bb  = draw.textbbox((0,0), title, font=FONT_LG)
    tw  = bb[2]-bb[0]
    draw.text((pw//2-tw//2, 10), title, fill=(r,g,b,255), font=FONT_LG)

    # Danger overlay
    if risk_val < SAFETY_THR:
        warn = '⚠ CRITICAL LOW STORAGE'
        bb   = draw.textbbox((0,0), warn, font=FONT_XL)
        tw,th = bb[2]-bb[0], bb[3]-bb[1]
        draw.text((pw//2-tw//2, ph//2-th//2), warn,
                  fill=(244,67,54,130), font=FONT_XL)

    return panel


# ── Build animation ───────────────────────────────────────────────────────────
print("Building animation frames with interpolation...")

def lerp(a, b, alpha):
    return a + alpha * (b - a)

# Pre-compute cumulative cost arrays for fast lookup
c_cumcost = np.cumsum(c_cost)
s_cumcost = np.cumsum(s_cost)

gif_frames = []

for fi, (t0, t1, alpha) in enumerate(interp_frames):
    if fi % 40 == 0:
        print(f"  Frame {fi+1}/{len(interp_frames)}  "
              f"(month {t0+1}, alpha={alpha:.2f})")

    # Interpolate all values
    t_float  = lerp(t0, t1, alpha)
    yr       = t_float / 12.0
    mo       = MONTHS[int(t_float) % 12]

    c_sc_f   = lerp(c_sc[t0],    c_sc[t1],    alpha) / SC_MAX
    c_sgi_f  = lerp(c_sgi[t0],   c_sgi[t1],   alpha) / SGI_MAX
    c_sw_f   = lerp(c_sswp[t0],  c_sswp[t1],  alpha) / SSWP_MAX
    c_des_f  = lerp(c_desal[t0], c_desal[t1], alpha) / desal_cap
    s_sc_f   = lerp(s_sc[t0],    s_sc[t1],    alpha) / SC_MAX
    s_sgi_f  = lerp(s_sgi[t0],   s_sgi[t1],   alpha) / SGI_MAX
    s_sw_f   = lerp(s_sswp[t0],  s_sswp[t1],  alpha) / SSWP_MAX
    s_des_f  = lerp(s_desal[t0], s_desal[t1], alpha) / desal_cap
    c_risk_v = lerp(c_risk[t0],  c_risk[t1],  alpha)
    s_risk_v = lerp(s_risk[t0],  s_risk[t1],  alpha)
    c_tot_m  = lerp(c_cumcost[t0], c_cumcost[t1], alpha) / 1e6
    s_tot_m  = lerp(s_cumcost[t0], s_cumcost[t1], alpha) / 1e6

    # Render panels
    left_panel  = render_panel(
        bg_left,  c_sc_f, c_sgi_f, c_sw_f, c_des_f,
        c_risk_v, c_tot_m, 'Cost-Only Agent', (33,150,243))
    right_panel = render_panel(
        bg_right, s_sc_f, s_sgi_f, s_sw_f, s_des_f,
        s_risk_v, s_tot_m, 'Safety-Constrained Agent', (76,175,80))

    # Compose frame
    frame = Image.new('RGBA', (FRAME_W, FRAME_H), (13,27,42,255))
    frame.paste(left_panel,  (0,       0))
    frame.paste(right_panel, (PANEL_W, 0))

    draw = ImageDraw.Draw(frame)

    # Divider
    draw.line([(PANEL_W,0),(PANEL_W,PANEL_H)],
              fill=(255,255,255,80), width=2)

    # Status bar
    bar_y = PANEL_H
    draw.rectangle([(0,bar_y),(FRAME_W,FRAME_H)],
                   fill=(10,15,30,255))

    # Title
    title = ('Santa Barbara Water Supply  |  '
             'RL Agent Comparison  |  Severe Drought Scenario')
    bb = draw.textbbox((0,0), title, font=FONT_MED)
    tw = bb[2]-bb[0]
    draw.text((FRAME_W//2-tw//2, bar_y+5), title,
              fill=(255,255,255,220), font=FONT_MED)

    # Date
    date_str = f'{mo}  |  Year {int(yr)+1}  (Month {int(t_float)+1} of {H})'
    bb = draw.textbbox((0,0), date_str, font=FONT_LG)
    tw = bb[2]-bb[0]
    draw.text((FRAME_W//2-tw//2, bar_y+26), date_str,
              fill=(255,255,255,255), font=FONT_LG)

    # Progress bar
    prog  = t_float / H
    bx0   = int(FRAME_W*0.1)
    bx1   = int(FRAME_W*0.9)
    byw   = bar_y + 56
    draw.rectangle([(bx0,byw),(bx1,byw+10)],
                   fill=(26,26,46,200), outline=(69,90,100,200))
    fill_x = int(bx0 + (bx1-bx0)*prog)
    if fill_x > bx0:
        draw.rectangle([(bx0,byw),(fill_x,byw+10)],
                       fill=(21,101,192,220))

    # Agent summary
    diff = s_tot_m - c_tot_m
    draw.text((10, bar_y+27),
              f'Cost-only: ${c_tot_m:,.0f}M  |  Risk: {c_risk_v:.1f} mo',
              fill=(33,150,243,230), font=FONT_SM)
    draw.text((10, bar_y+43),
              f'Safe:      ${s_tot_m:,.0f}M  (+${diff:,.0f}M)  |  Risk: {s_risk_v:.1f} mo',
              fill=(76,175,80,230), font=FONT_SM)

    # Safety note
    note = f'Safety threshold: {SAFETY_THR} months'
    bb   = draw.textbbox((0,0), note, font=FONT_SM)
    tw   = bb[2]-bb[0]
    draw.text((FRAME_W-tw-10, bar_y+43), note,
              fill=(244,67,54,200), font=FONT_SM)

    gif_frames.append(frame.convert('RGB'))

# ── Save GIF ──────────────────────────────────────────────────────────────────
out_path = os.path.join(ANIM_DIR, 'map_animation.gif')
print(f"\nSaving {len(gif_frames)}-frame GIF → {out_path}")
gif_frames[0].save(
    out_path,
    save_all      = True,
    append_images = gif_frames[1:],
    duration      = 80,
    loop          = 0,
    optimize      = False,
)
print(f"✅ Done → {out_path}")