import pandas as pd
import re
import os
import base64
import torch
import pymongo
from bs4 import BeautifulSoup
import subprocess
import tempfile
from io import StringIO
from django.conf import settings
import logging
import fitz
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from typing import List, Dict, Optional
import json
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import time
from chatbot.offline_loader import *
from transformers import GPT2TokenizerFast

def parse_field(field_str):
    """
    Parse a field definition string like 'BUFFER_SIZE [15:0]' to extract:
    - Field name
    - LSB position
    - MSB position
    - Field size
    """
    if isinstance(field_str, str):
        # Match field name and index range, e.g., 'BUFFER_SIZE [15:0]'
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[(\d+):(\d+)\])?", field_str)
        if match:
            field_name = match.group(1).strip()
            msb = int(match.group(2)) if match.group(2) else 0  # Handle case where range is not provided
            lsb = int(match.group(3)) if match.group(3) else 0
            size = msb - lsb + 1 if msb and lsb else 1  # Default to size 1 if no range
            return field_name, size, lsb, msb
    return None, None, None, None

def excel_to_uvm_ral(excel_file, output_file):
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file, header=None)

        # Strip all column names of leading/trailing whitespace
        df.columns = df.iloc[0].str.strip()
        df = df[1:]

        # Check if the necessary columns are present
        required_columns = ['Register Name', 'Offset', 'Read/Write', 'Fields', 'Default value', 'Reset value', 'Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            return

        # Map column names to expected names
        column_map = {
            'Register Name': 'register_name',
            'Offset': 'offset',
            'Read/Write': 'read_write',
            'Fields': 'fields',
            'Default value': 'default_value',
            'Reset value': 'reset_value',
            'Description': 'description'
        }
        df.rename(columns=column_map, inplace=True)

        # Convert all relevant columns to strings and fill NaN with default values
        df['register_name'] = df['register_name'].fillna("").astype(str)
        df['fields'] = df['fields'].fillna("").astype(str)

        # Open the output file for writing
        with open(output_file, 'w') as f:
            # Write the UVM RAL header
            f.write("`ifndef REG_MODEL\n")
            f.write("`define REG_MODEL\n\n")

            current_reg = None
            fields = []

            for _, row in df.iterrows():
                reg_name = row['register_name'].strip()
                reg_offset = row['offset']
                read_write = row['read_write']
                fields_str = row['fields']
                default_value = row['default_value']
                reset_value = row['reset_value']
                description = row['description']

                # If we encounter a new register, write out the previous register and its fields
                if reg_name:  # Register name found
                    # If we're already processing a register, close it out
                    if current_reg:
                        # Write the previous register's field instances (before constructor)
                        f.write(f"  //---------------------------------------\n")
                        for field in fields:
                            field_name, size, lsb, msb = parse_field(field.strip())
                            if field_name and size and lsb is not None:
                                f.write(f"    rand uvm_reg_field {field_name};\n")
                        
                        # Write the build_phase only once
                        f.write(f"  //---------------------------------------\n")
                        f.write("  function void build;\n")
                        for field in fields:  # Indentation fixed here
                            field_name, size, lsb, msb = parse_field(field.strip())
                            if field_name and size and lsb is not None:
                               f.write(f"    {field_name} = uvm_reg_field::type_id::create(\"{field_name}\");\n")
                               f.write(f"    {field_name}.configure(.parent(this),\n")
                               f.write(f"                           .size({size}),\n")
                               f.write(f"                           .lsb_pos({lsb}),\n")
                               f.write(f"                           .msb_pos({msb}),\n")
                               f.write(f"                           .access(\"{read_write}\"),\n")
                               f.write(f"                           .volatile(0),\n")
                               f.write(f"                           .reset({default_value if pd.notna(default_value) else 0}),\n")
                               f.write(f"                           .has_reset(1),\n")
                               f.write(f"                           .is_rand(1),\n")
                               f.write(f"                           .individually_accessible(0));\n")
                        f.write("  endfunction\n")
                        f.write("endclass\n\n")

                    # Now process the new register
                    f.write(f"class {reg_name} extends uvm_reg;\n")
                    f.write(f"  `uvm_object_utils({reg_name})\n\n")
                    f.write("  //---------------------------------------\n")

                    # Initialize new fields list
                    fields = []
                    current_reg = reg_name

                    # Write register constructor (only once)
                    f.write(f"  // Constructor\n")
                    f.write(f"  //---------------------------------------\n")
                    f.write(f"  function new(string name = \"{reg_name}\");\n")
                    f.write(f"    super.new(name, 32, UVM_NO_COVERAGE);\n")
                    f.write(f"  endfunction\n\n")
                # Add fields to the list
                if fields_str:  # Process each field for the current register
                    fields.append(fields_str.strip())

            # After the last register, write it out
            if current_reg:
                # Write the last register's field instances and configuration
                f.write(f"  //---------------------------------------\n")
                for field in fields:
                    field_name, size, lsb, msb = parse_field(field.strip())
                    if field_name and size and lsb is not None:
                        f.write(f"    rand uvm_reg_field {field_name};\n")
                        f.write(f"    {field_name} = uvm_reg_field::type_id::create(\"{field_name}\");\n")
                        f.write(f"    {field_name}.configure(.parent(this),\n")
                        f.write(f"                           .size({size}),\n")
                        f.write(f"                           .lsb_pos({lsb}),\n")
                        f.write(f"                           .msb_pos({msb}),\n")
                        f.write(f"                           .access(\"{read_write}\"),\n")
                        f.write(f"                           .volatile(0),\n")
                        f.write(f"                           .reset({default_value if pd.notna(default_value) else 0}),\n")
                        f.write(f"                           .has_reset(1),\n")
                        f.write(f"                           .is_rand(1),\n")
                        f.write(f"                           .individually_accessible(0));\n")
                f.write("  endfunction\n")
                f.write("endclass\n\n")

            # Generate the register block class
            f.write("//-------------------------------------------------------------------------\n")
            f.write("//\tRegister Block Definition\n")
            f.write("//-------------------------------------------------------------------------\n")
            f.write("class dma_reg_model extends uvm_reg_block;\n")
            f.write("  `uvm_object_utils(dma_reg_model)\n\n")
            f.write("  //---------------------------------------\n")
            f.write("  // Register Instances\n")
            f.write("  //---------------------------------------\n")

            for reg_name in df['register_name'].dropna().unique():
                f.write(f"  rand {reg_name} reg_{reg_name.lower()};\n")

            f.write("\n  //---------------------------------------\n")
            f.write("  // Constructor\n")
            f.write("  //---------------------------------------\n")
            f.write("  function new (string name = \"\");\n")
            f.write("    super.new(name, build_coverage(UVM_NO_COVERAGE));\n")
            f.write("  endfunction\n\n")
            f.write("  //---------------------------------------\n")
            f.write("  // Build Phase\n")
            f.write("  //---------------------------------------\n")
            f.write("  function void build();\n")

            for reg_name in df['register_name'].dropna().unique():
                f.write(f"    reg_{reg_name.lower()} = {reg_name}::type_id::create(\"reg_{reg_name.lower()}\");\n")
                f.write(f"    reg_{reg_name.lower()}.build();\n")
                f.write(f"    reg_{reg_name.lower()}.configure(this);\n")

            f.write("    //---------------------------------------\n")
            f.write("    // Memory Map Creation and Register Map\n")
            f.write("    //---------------------------------------\n")
            f.write("    default_map = create_map(\"my_map\", 0, 4, UVM_LITTLE_ENDIAN);\n")
            for reg_name in df['register_name'].dropna().unique():
                f.write(f"    default_map.add_reg(reg_{reg_name.lower()}, 'h0, \"RW\");\n")

            f.write("    lock_model();\n")
            f.write("  endfunction\n")
            f.write("endclass\n\n")

            f.write("`endif // REG_MODEL\n")

        print(f"UVM RAL file generated: {output_file}")

    except Exception as e:
        print(f"Error: {e}")


# Hide llama.cpp logs
logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["Verif_Playground"]
scripts_collection = db.get_collection("scripts")

# ✅ Offline Embedding Model
model = OfflineSentenceTransformerEmbeddings()

# Text extractors
def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""

def is_base64(text):
    clean = re.sub(r"<[^>]+>", "", text).strip().replace("\n", "").replace(" ", "")
    if len(clean) < 100:
        return False
    try:
        base64.b64decode(clean, validate=True)
        return True
    except Exception:
        return False

def decode_base64(text):
    try:
        padded = text + "=" * (-len(text) % 4)
        return base64.b64decode(padded)
    except Exception:
        return None

def process_mongodb_document(doc: Dict) -> Optional[Document]:
    file_type = doc.get("fileType", "").lower()
    html_data = doc.get("htmlData", "")
    base64_data = doc.get("base64", "")
    text = ""

    try:
        if file_type == "html":
            if html_data:
                if is_base64(html_data):
                    decoded = decode_base64(html_data)
                    if decoded:
                        text = extract_text_from_html(decoded.decode("utf-8", errors="ignore"))
                else:
                    text = extract_text_from_html(html_data)

        elif file_type == "pdf" and base64_data:
            if is_base64(base64_data):
                decoded = decode_base64(base64_data)
                if decoded:
                    text = extract_text_from_pdf_bytes(decoded)

        if not text.strip():
            return None

        return Document(
            page_content=text,
            metadata={
                "_id": str(doc.get("_id")),
                "fileName": doc.get("fileName", "unknown"),
                "fileType": file_type
            }
        )
    except Exception as e:
        print(f"Error processing document {doc.get('_id')}: {str(e)}")
        return None

# ⏱️ Vector DB Initialization
def initialize_mongodb_vector_db():
    embedding_path = "chatbot/vector_data"

    try:
        start_time = time.time()
        if os.path.exists(os.path.join(embedding_path, "chroma-collections.parquet")):
            vector_db = Chroma(
                persist_directory=embedding_path,
                embedding_function=model
            )
        else:

            documents = []
            for doc in scripts_collection.find():
                processed = process_mongodb_document(doc)
                if processed:
                    documents.append(processed)

            if not documents:
                raise ValueError("❌ No valid documents found in MongoDB")

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_documents = splitter.split_documents(documents)

            vector_db = Chroma.from_documents(
                documents=split_documents,
                embedding=model,
                persist_directory=embedding_path
            )

        elapsed = time.time() - start_time
        if elapsed > 180:
            print("⚠️ WARNING: Vector loading exceeded 3 minutes. NGINX may timeout!")

        return vector_db.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        print(f"❌ Error initializing vector DB: {str(e)}")
        return None

# LLM Initialization
def initialize_llm():
    try:
        model_path = os.path.join("chatbot/models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        return Llama(
            model_path=model_path,
            n_ctx=32768,
            n_threads=4,
            use_mlock=True,
            verbose=False
        )
    except Exception as e:
        print(f"❌ Error initializing LLM: {str(e)}")
        return None

# Load on module import
try:
    retriever = initialize_mongodb_vector_db()
    llm = initialize_llm()
except Exception as e:
    print(f"❌ Error initializing chatbot components: {str(e)}")
    retriever = None
    llm = None


# Chat function
# def get_chatbot_response(query: str) -> str:

#     if not retriever:
#         return "Chatbot service is currently unavailable. Please try again later."

#     if not llm:
#         return "Chatbot service is currently unavailable. Please try again later."

#     try:
#         docs = retriever.get_relevant_documents(query)

#         if not docs or all(not doc.page_content.strip() for doc in docs):
#             return "Sorry, I'm not trained for this specific topic."

#         context = "\n".join([doc.page_content for doc in docs])

#         prompt = f"""You are a helpful assistant. Answer using only this context.
# If the answer isn't here, say "I'm not trained for this."

# Context:
# {context}

# Question: {query}

# Plain text answer:"""

#         response = llm(prompt, max_tokens=512, stop=["</s>"])

#         answer = response["choices"][0]["text"].strip()

#         return answer if answer else "Sorry, I couldn't generate a response."

#     except Exception as e:
#         print(f"Error generating response: {str(e)}")
#         return "An error occurred while processing your request."


def get_chatbot_response(query: str) -> str:
    if not retriever:
        return "Chatbot service is currently unavailable. Please try again later."

    if not llm:
        return "Chatbot service is currently unavailable. Please try again later."

    try:
        # Get relevant documents with increased k value to get more context
        docs = retriever.get_relevant_documents(query)
        
        if not docs or all(not doc.page_content.strip() for doc in docs):
            return "Sorry, I couldn't find specific information about this topic in my knowledge base. " \
                   "I can provide general verification procedures if you'd like."

        # Enhanced context preparation
        context = ""
        for i, doc in enumerate(docs, 1):
            context += f"\n\nDOCUMENT {i} (Source: {doc.metadata.get('fileName', 'unknown')}):\n"
            context += doc.page_content
        
        # Enhanced prompt template
        prompt = f"""You are a verification engineering assistant. Provide detailed, step-by-step explanations 
and comprehensive procedures when answering questions. Include relevant examples and considerations where appropriate.

When responding:
1. Do NOT invent content.
2. Explain terms exactly as written in the documents.
3. Keep the structure and wording close to the original when helpful.
4. Provide structured answers (definition → explanation → conclusion).
5. Conclude with summary

If the question is unclear or you need more information, ask clarifying questions.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

DETAILED ANSWER:"""

        # Generate response with higher token limit for more detailed answers
        response = llm(
            prompt,
            max_tokens=1024,  # Increased from 512 for more detailed responses
            temperature=0.3,  # Lower temperature for more focused answers
            stop=["</s>"]
        )

        answer = response["choices"][0]["text"].strip()

        if not answer:
            return "I couldn't generate a response. Please try rephrasing your question or ask about a different topic."
        
        # Post-process the answer to ensure it meets our quality standards
        if len(answer.split()) < 50:  # If answer is too short
            return f"I found some information that might be relevant:\n\n{context}\n\n" \
                   "Would you like me to elaborate on any specific part of this information?"
        
        return answer

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "An error occurred while processing your request. Please try again later."

def run_mux_simulation(design_file, tb_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        design_path = os.path.join(temp_dir, "mux_design.sv")
        tb_path = os.path.join(temp_dir, "mux_tb.py")
        makefile_path = os.path.join(temp_dir, "Makefile")
        excel_output_path = os.path.join(temp_dir, "mux_result.xlsx")
        vcd_output_path = os.path.join(temp_dir, "dump.vcd")

        # Output files to be saved inside settings.MEDIA_ROOT
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        excel_copy_path = os.path.join(settings.MEDIA_ROOT, "mux_simulation_result.xlsx")
        vcd_copy_path = os.path.join(settings.MEDIA_ROOT, "mux_dump.vcd")

        # Save uploaded files
        with open(design_path, 'wb') as f:
            f.write(design_file.read())
        with open(tb_path, 'wb') as f:
            f.write(tb_file.read())

        # Create Makefile with just filenames (not full paths)
        makefile_content = f"""
TOPLEVEL_LANG ?= verilog
SIM ?= icarus
VERILOG_SOURCES = mux_design.sv
TOPLEVEL := mux
MODULE := mux_tb

include $(shell cocotb-config --makefiles)/Makefile.sim
"""
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)

        # Convert Windows path to WSL format: C:\Users\hi\... -> /mnt/c/Users/hi/...
        wsl_path = temp_dir.replace("\\", "/").replace(":", "")
        wsl_path = "/mnt/" + wsl_path[0].lower() + wsl_path[1:]

        # Correct subprocess call: export PATH and run inside bash correctly
        result = subprocess.run(
            ["wsl", "bash", "-c", f"export PATH=\"$HOME/.local/bin:$PATH\" && cd '{wsl_path}' && make sim=icarus"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return {"error": "Simulation failed", "output": result.stderr}

        output = result.stdout

        # Parse lines like: a: 44 b: 177 c: 95 d: 253 sel: 3 dout: 253
        pattern = re.compile(r"a: (\d+) b: (\d+) c: (\d+) d: (\d+) sel: (\d+) dout: (\d+)")
        rows = pattern.findall(output)

        if not rows:
            return {"error": "No waveform-style data found in simulation output", "output": output}

        df = pd.DataFrame(rows, columns=["a", "b", "c", "d", "sel", "dout"])
        df.to_excel(excel_output_path, index=False)

        # Save Excel to media folder
        with open(excel_output_path, 'rb') as fsrc, open(excel_copy_path, 'wb') as fdst:
            fdst.write(fsrc.read())

        # Copy VCD to media folder
        if os.path.exists(vcd_output_path):
            with open(vcd_output_path, 'rb') as fsrc, open(vcd_copy_path, 'wb') as fdst:
                fdst.write(fsrc.read())

        return {
            "message": "Simulation successful",
            "excel_file": settings.MEDIA_URL + "mux_simulation_result.xlsx",
            "vcd_file": settings.MEDIA_URL + "mux_dump.vcd",
            "stdout": output,
            "df": df
        }

def generate_waveform_from_excel(file_obj):
    """
    Generate waveform.json, waveform.png, waveform.svg, and styled PNG from an uploaded Excel file.
    Returns the styled PNG path or an error.
    """
    try:
        # Save uploaded file to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            input_excel = os.path.join(temp_dir, "input.xlsx")
            with open(input_excel, 'wb') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)

            df = pd.read_excel(input_excel, engine="openpyxl")
            data_buses = list(df.columns)

            # --- Inline convert_to_wavejson ---

            wavejson = {"signal": []}
            num_timepoints = len(df)
            for bus in data_buses:
                signal_line = {"name": bus, "wave": ""}
                values = [str(v).strip().upper() for v in df[bus].tolist()]
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
                    signal_line["data"] = [
                        d for w, d in zip(signal_line["wave"], wave_data) if w == '=' and d is not None
                    ]
                wavejson["signal"].append(signal_line)
            # -----------------------------------

            json_path = os.path.join(temp_dir, "waveform.json")
            png_path = os.path.join(temp_dir, "waveform.png")
            svg_path = os.path.join(temp_dir, "waveform.svg")

            with open(json_path, "w") as f:
                json.dump(wavejson, f, indent=2)

            wavedrom_path = shutil.which("wavedrom-cli")
            if not wavedrom_path:
                return {"error": "wavedrom-cli not found in PATH."}

            subprocess.run([wavedrom_path, "-i", json_path, "-p", png_path], check=True, shell=True)
            subprocess.run([wavedrom_path, "-i", json_path, "-p", svg_path], check=True, shell=True)

            # --- Inline png_post_process ---
            styled_png_path = os.path.join(settings.MEDIA_ROOT, "waveform_styled.png")
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            try:
                img = Image.open(png_path).convert("RGBA")
                w, h = img.size
                scale = 1 if w > 1500 else 2
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
                    shade = 245 - int(25 * (y / canvas_h))
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

                gradient_img.save(styled_png_path, optimize=True)
            except Exception as e:
                return {"error": f"Post-process styling failed: {e}"}
            # -----------------------------------

            return {
                "message": "Waveform generated successfully",
                "image_url": settings.MEDIA_URL + "waveform_styled.png"
            }

    except subprocess.CalledProcessError as e:
        return {"error": f"Error generating waveform image: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
