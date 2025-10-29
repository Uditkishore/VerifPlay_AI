#run "npm install -g wavedrom-cli" to install wavedrom-cli
#run "pip install pandas openpyxl" to install required packages 
#please ensure wavedrom-cli is in your PATH, you can use "C:\Users\YourUsername\AppData\Roaming\npm" for Windows or "/usr/local/bin" for Linux/MacOS

import pandas as pd
import json
import subprocess
import os
import shutil

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
                if val == last_val:
                    signal_line["wave"] += "."
                else:
                    signal_line["wave"] += val
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
                if last_val == val:
                    signal_line["wave"] += "."
                    wave_data.append(None)
                else:
                    signal_line["wave"] += "="
                    wave_data.append(val)
                    last_val = val

        assert len(signal_line["wave"]) == num_timepoints, f"{bus} has mismatch in waveform length"

        if '=' in signal_line["wave"]:
            signal_line["data"] = [d for w, d in zip(signal_line["wave"], wave_data) if w == '=' and d is not None]

        wavejson["signal"].append(signal_line)

    return wavejson


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "temp_test/Extended_Waveform_Andgate.xlsx")
    output_json = os.path.join(script_dir, "waveform.json")
    output_png = os.path.join(script_dir, "waveform.png")
    output_svg = os.path.join(script_dir, "waveform.svg")

    try:
        # ✅ Load Excel and infer headers as data buses
        data = pd.read_excel(file_path, engine="openpyxl")
        data_buses = list(data.columns)
        print(f"✅ Detected data buses: {data_buses}")

        wavejson = convert_to_wavejson(data, data_buses)

        with open(output_json, "w") as f:
            json.dump(wavejson, f, indent=2)
        print("✅ waveform.json created successfully!")

        wavedrom_path = shutil.which("wavedrom-cli")
        if not wavedrom_path:
            raise FileNotFoundError("wavedrom-cli not found in PATH. Please install it and check your environment.")

        print(f"wavedrom-cli found at: {wavedrom_path}")

        # ✅ Generate PNG instead of SVG
        subprocess.run(
            [wavedrom_path, "-i", output_json, "-p", output_png],
            check=True,
            shell=True
        )
        subprocess.run(
            [wavedrom_path, "-i", output_json, "-p", output_svg],
            check=True,
            shell=True
        )
        print(f"✅ waveform.png created successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating PNG: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
