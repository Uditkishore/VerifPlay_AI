import os
import json
import shutil
import subprocess
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np


def convert_to_wavejson(data, data_buses):
    wavejson = {"signal": []}
    num_timepoints = len(data)
    for bus in data_buses:
        signal_line = {"name": bus, "wave": ""}
        values = [str(v).strip().upper() for v in data[bus].tolist()]
        wave_data = []
        last_val = None
        for val in values:
            if val in {"0", "1"}:
                signal_line["wave"] += "." if val == last_val else val
                wave_data.append(None)
                last_val = val
            elif val == "X":
                signal_line["wave"] += "x"
                wave_data.append(None)
                last_val = val
            elif val == "Z":
                signal_line["wave"] += "z"
                wave_data.append(None)
                last_val = val
            else:
                signal_line["wave"] += "." if val == last_val else "="
                wave_data.append(None if val == last_val else val)
                last_val = val
        if '=' in signal_line["wave"]:
            signal_line["data"] = [d for w, d in zip(signal_line["wave"], wave_data) if w == '=' and d is not None]
        wavejson["signal"].append(signal_line)
    return wavejson

def png_post_process(png_path, out_path=None):
    try:
        if out_path is None:
            root, ext = os.path.splitext(png_path)
            out_path = f"{root}_styled{ext}"
        img = Image.open(png_path).convert("RGBA")
        w, h = img.size
        

        SCALE_LIMIT = 1500
        scale = 1 if w > SCALE_LIMIT else 2
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

        shadow_offset = (12, 12)
        blur_radius = 14
        alpha = img.split()[-1]
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 180))
        shadow.putalpha(alpha)
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

        pad = 24
        canvas_w = min(img.width + pad * 2 + shadow_offset[0], 3000)
        canvas_h = min(img.height + pad * 2 + shadow_offset[1] + 60, 3000)

        gradient_img = Image.new("RGBA", (canvas_w, canvas_h), (240, 240, 240, 255))

        draw_bg = ImageDraw.Draw(gradient_img)
        for y in range(canvas_h):
            shade = 245 - int(25 * (y / canvas_h))  # from 245 to 220
            draw_bg.line([(0, y), (canvas_w, y)], fill=(shade, shade, shade, 255))

        gradient_img.paste(shadow, (pad + shadow_offset[0], pad + shadow_offset[1]), shadow)
        gradient_img.paste(img, (pad, pad), img)

        draw = ImageDraw.Draw(gradient_img)
        banner_h = 48
        banner_rect = [0, 0, canvas_w, banner_h]
        draw.rectangle(banner_rect, fill=(30, 41, 59, 255))

        font_size = 28
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text = "Digital Waveform"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.text(((canvas_w - tw) / 2, (banner_h - th) / 2 - 2), text, font=font, fill=(255, 255, 255, 255))
        
        wm_text = "© MyEDA Tool"
        wm_font = font if font != ImageFont.load_default() else ImageFont.load_default()
        wm_bbox = draw.textbbox((0, 0), wm_text, font=wm_font)
        wmw = wm_bbox[2] - wm_bbox[0]
        wmh = wm_bbox[3] - wm_bbox[1]
        draw.text((canvas_w - wmw - 8, canvas_h - wmh - 6), wm_text, font=wm_font, fill=(0, 0, 0, 100))

        gradient_img.save(out_path, optimize=True)
        print(f"✅ Styled PNG saved → {out_path}")
    except Exception as e:
        print(f"❌ Post-process styling failed: {e}")


if __name__ == "__main__":
    print("🚀 Script started...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Extended_Waveform_Andgate.xlsx")
    output_json = os.path.join(script_dir, "waveform.json")
    output_png = os.path.join(script_dir, "waveform.png")
    output_svg = os.path.join(script_dir, "waveform.svg")

    try:
        data = pd.read_excel(file_path, engine="openpyxl")
        data_buses = list(data.columns)
        print(f"✅ Detected data buses: {data_buses}")
        wavejson = convert_to_wavejson(data, data_buses)
        with open(output_json, "w") as f:
            json.dump(wavejson, f, indent=2)
        print("✅ waveform.json created successfully!")

        wavedrom_path = shutil.which("wavedrom-cli")
        if not wavedrom_path:
            raise FileNotFoundError("❌ wavedrom-cli not found in PATH.")
        print(f"✅ wavedrom-cli found at: {wavedrom_path}")

        subprocess.run([wavedrom_path, "-i", output_json, "-p", output_png], check=True, shell=True)
        subprocess.run([wavedrom_path, "-i", output_json, "-p", output_svg], check=True, shell=True)
        print("✅ waveform.png and waveform.svg created successfully!")

        png_post_process(output_png)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running wavedrom-cli: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

