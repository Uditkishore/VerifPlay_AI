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
from typing import List, Dict, Optional, Any, Tuple
import json
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import time
from chatbot.offline_loader import *
from transformers import GPT2TokenizerFast
import tiktoken
from datetime import datetime
import difflib

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
    
def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return 0


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
        model_path = "chatbot/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
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
    print("✅ Chatbot models loaded successfully!")
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
        prompt_tokens = count_tokens(prompt)
        # max_allowed_tokens = 2000  # your total limit (safe for llama-cpp)
        max_gen_tokens = prompt_tokens

        # Ensure positive value
        if max_gen_tokens < 200:
            max_gen_tokens = 200

        print(f"[TOKEN DEBUG] Prompt tokens = {prompt_tokens}, Generating = {max_gen_tokens}")

        response = llm(
            prompt,
            # max_tokens=1024,  # Increased from 512 for more detailed responses
            max_tokens=max_gen_tokens,
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


# In-memory store for uploaded files and derived state
_STORE: Dict[str, Dict[str, Any]] = {}


class VerilogBackend:
    def __init__(self, store: Dict[str, Dict[str, Any]] = None, persist_folder: Optional[str] = None):
        self.store = store if store is not None else _STORE
        self.persist_folder = persist_folder
        if self.persist_folder:
            os.makedirs(self.persist_folder, exist_ok=True)

    # -----------------------------
    # File store helpers
    # -----------------------------
    def save_uploaded_file(self, filename: str, data: bytes) -> str:
        """Save uploaded file bytes -> store key"""
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = data.decode("latin1", errors="ignore")
        else:
            text = str(data)

        key = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{os.path.basename(filename)}"
        self.store[key] = {
            "filename": filename,
            "content": text,
            "saved_at": datetime.utcnow().isoformat(),
            # caches / editor state:
            "highlight_html": None,
            "last_explanation": None
        }

        if self.persist_folder:
            path = os.path.join(self.persist_folder, key)
            with open(path, "w", encoding="utf-8", errors="ignore") as f:
                f.write(text)

        return key

    def get_saved_file(self, key: str) -> Optional[str]:
        it = self.store.get(key)
        return it.get("content") if it else None

    # -----------------------------
    # Resolve code input
    # -----------------------------
    def _resolve(self, source_or_key: str) -> str:
        if not source_or_key:
            return ""
        if source_or_key in self.store:
            return self.store[source_or_key]["content"]
        return str(source_or_key)

    # -----------------------------
    # Explain / parse / report
    # -----------------------------
    def explain_code(self, source_or_key: str) -> Dict[str, Any]:
        code = self._resolve(source_or_key).strip()
        if not code:
            return {"status": "error", "message": "No code provided", "explanation": ""}

        modules = self._fast_extract_modules_optimized(code)
        explanation = self._generate_explanation_optimized(code, modules)
        # cache last explanation if key
        if source_or_key in self.store:
            self.store[source_or_key]["last_explanation"] = explanation

        return {"status": "ok", "module_count": len(modules), "modules": [{"name": m["name"]} for m in modules],
                "explanation": explanation}

    def generate_testbench(self, source_or_key: str, mode: str = "auto") -> Dict[str, Any]:
        code = self._resolve(source_or_key)
        modules = self._fast_extract_modules_optimized(code)
        if not modules:
            return {"status": "error", "message": "No modules found", "testbench": ""}

        info = self._extract_detailed_module_info(code) or {"module_name": modules[0]["name"], "inputs": [], "outputs": [], "inouts": [], "parameters": []}
        is_apb, signals = self._detect_apb_protocol(code[:20000])
        if mode == "apb" or (mode == "auto" and is_apb):
            tb = self._build_apb_testbench(info, signals)
            kind = "apb"
        else:
            tb = self._build_comprehensive_testbench(info)
            kind = "simple"
        return {"status": "ok", "module": info.get("module_name"), "type": kind, "testbench": tb}

    def generate_uvm_testbench(self, source_or_key: str) -> Dict[str, Any]:
        code = self._resolve(source_or_key)
        info = self._extract_detailed_module_info(code)
        if not info:
            modules = self._fast_extract_modules_optimized(code)
            if not modules:
                return {"status": "error", "message": "No modules found", "testbench": ""}
            info = {"module_name": modules[0]["name"], "inputs": [], "outputs": [], "inouts": [], "parameters": []}
        tb = self._build_uvm_testbench(info)
        return {"status": "ok", "module": info.get("module_name"), "testbench": tb}

    def generate_design_report(self, source_or_key: str) -> Dict[str, Any]:
        code = self._resolve(source_or_key)
        modules = self._fast_extract_modules_optimized(code)
        sample = code[:200000]
        total_lines = len(code.splitlines())
        always_count = len(re.findall(r'always\s*@', sample))
        assign_count = len(re.findall(r'\bassign\b', sample))
        instance_count = len(re.findall(r'(\w+)\s+(\w+)\s*\(', sample))
        module_summaries = []
        for m in modules[:50]:
            insts = re.findall(r'\b(\w+)\s+(\w+)\s*\(', m["text"][:20000])
            module_summaries.append({"name": m["name"], "instances": len(insts), "snippet": m["text"][:400]})
        top = max(module_summaries, key=lambda x: x["instances"])["name"] if module_summaries else None
        return {
            "status": "ok",
            "total_lines": total_lines,
            "module_count": len(modules),
            "instance_count_sampled": instance_count,
            "assign_count_sampled": assign_count,
            "always_count_sampled": always_count,
            "top_module_candidate": top,
            "modules": module_summaries
        }

    # -----------------------------
    # Copy / Clear / Append / Chunks / Editor helpers
    # -----------------------------
    def copy_content(self, source_or_key: str, kind: str = "code") -> Dict[str, Any]:
        """
        Return the requested content to be 'copied' by client.
        kind: 'code' or 'explanation' or 'testbench'
        """
        if kind == "explanation":
            if source_or_key in self.store and self.store[source_or_key].get("last_explanation"):
                return {"status": "ok", "content": self.store[source_or_key]["last_explanation"]}
            else:
                # generate on-the-fly
                res = self.explain_code(source_or_key)
                return {"status": res.get("status", "error"), "content": res.get("explanation", "")}
        elif kind == "testbench":
            res = self.generate_testbench(source_or_key)
            return {"status": res.get("status", "error"), "content": res.get("testbench", "")}
        else:
            # kind == 'code' default
            content = self._resolve(source_or_key)
            return {"status": "ok" if content else "error", "content": content}

    def clear_all(self) -> Dict[str, Any]:
        """Clear the in-memory store entirely."""
        self.store.clear()
        return {"status": "ok", "message": "Store cleared"}

    def add_chunk(self, key: str, chunk_text: str) -> Dict[str, Any]:
        """Append chunk_text to an existing stored file."""
        if key not in self.store:
            return {"status": "error", "message": "Key not found"}
        self.store[key]["content"] += str(chunk_text)
        if self.persist_folder:
            try:
                path = os.path.join(self.persist_folder, key)
                with open(path, "a", encoding="utf-8", errors="ignore") as f:
                    f.write(str(chunk_text))
            except Exception:
                pass
        return {"status": "ok", "new_length": len(self.store[key]["content"])}

    def apply_next_chunk(self, key: str, offset: int = 0, chunk_size: int = 10000) -> Dict[str, Any]:
        """Return the next chunk from stored content starting at offset. Useful for streaming UI."""
        content = self.get_saved_file(key)
        if content is None:
            return {"status": "error", "message": "Key not found"}
        total = len(content)
        if offset >= total:
            return {"status": "ok", "chunk": "", "next_offset": total, "done": True}
        end = min(total, offset + chunk_size)
        chunk = content[offset:end]
        done = end >= total
        return {"status": "ok", "chunk": chunk, "next_offset": end, "done": done, "total": total}

    def update_line_numbers(self, source_or_key: str) -> Dict[str, Any]:
        """Return the count of lines and a small preview of numbered lines."""
        content = self._resolve(source_or_key)
        lines = content.splitlines()
        count = len(lines)
        preview_lines = []
        max_preview = 10
        for i in range(min(count, max_preview)):
            preview_lines.append({"ln": i + 1, "text": lines[i][:200]})
        return {"status": "ok", "lines": count, "preview": preview_lines}

    def on_code_change(self, key: str, new_code: str) -> Dict[str, Any]:
        """
        Update stored code and return a small unified diff preview.
        If key not present, create a new stored item with filename=key.
        """
        old = self.get_saved_file(key) or ""
        self.store.setdefault(key, {"filename": key, "content": "", "saved_at": datetime.utcnow().isoformat()})
        self.store[key]["content"] = new_code
        self.store[key]["saved_at"] = datetime.utcnow().isoformat()
        # compute small diff
        old_lines = old.splitlines(keepends=False)
        new_lines = new_code.splitlines(keepends=False)
        diff = list(difflib.unified_diff(old_lines[:200], new_lines[:200], lineterm=""))
        diff_preview = "\n".join(diff[:200])
        return {"status": "ok", "diff_preview": diff_preview}

    def find_text(self, source_or_key: str, query: str, max_results: int = 50) -> Dict[str, Any]:
        """Return occurrences (line numbers + small context) for query string or regex."""
        content = self._resolve(source_or_key)
        if not content:
            return {"status": "error", "message": "No content"}
        lines = content.splitlines()
        results = []
        pattern = None
        is_regex = False
        try:
            pattern = re.compile(query)
            is_regex = True
        except re.error:
            pattern = None
            is_regex = False

        for idx, line in enumerate(lines, start=1):
            if is_regex:
                if pattern.search(line):
                    results.append({"line": idx, "text": line.strip()})
            else:
                if query in line:
                    results.append({"line": idx, "text": line.strip()})
            if len(results) >= max_results:
                break
        return {"status": "ok", "occurrences": len(results), "results": results}

    # -----------------------------
    # Syntax highlighting (backend HTML)
    # -----------------------------
    def highlight_code(self, source_or_key: str) -> Dict[str, Any]:
        """Return a simple HTML highlighted version of the code (not exhaustive)."""
        code = self._resolve(source_or_key)
        if not code:
            return {"status": "error", "html": ""}

        html = self._highlight_syntax(code)
        # cache if key
        if source_or_key in self.store:
            self.store[source_or_key]["highlight_html"] = html
        return {"status": "ok", "html": html}

    def _highlight_syntax(self, code: str) -> str:
        """
        Very small syntax highlighter for Verilog-like code.
        Returns HTML string with <span class="kw">..</span> etc.
        """
        # escape HTML
        esc = lambda s: (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

        # simple tokenization: comments, strings, keywords, numbers
        # 1) protect multiline comments and single-line
        code_escaped = esc(code)

        # mark multiline comments
        code_escaped = re.sub(r'(/\*.*?\*/)', r'<span class="comment">\1</span>', code_escaped, flags=re.DOTALL)
        # single line comments
        code_escaped = re.sub(r'(//.*?$)', r'<span class="comment">\1</span>', code_escaped, flags=re.MULTILINE)

        # strings
        code_escaped = re.sub(r'(\".*?\")', r'<span class="str">\1</span>', code_escaped)

        # keywords (a small set)
        keywords = r'\b(module|endmodule|input|output|inout|wire|reg|logic|always|assign|if|else|begin|end|parameter|module)\b'
        code_escaped = re.sub(keywords, r'<span class="kw">\1</span>', code_escaped)

        # numbers
        code_escaped = re.sub(r'\b(\d+\'[bdhBDH]?[0-9a-fA-FxXzZ_]+|\d+)\b', r'<span class="num">\1</span>', code_escaped)

        # wrap in pre
        styles = (
            "<style>"
            ".kw{color:#d73a49;font-weight:600} "
            ".comment{color:#6a9955} "
            ".str{color:#032f62} "
            ".num{color:#005cc5} "
            "pre{white-space:pre-wrap;font-family:Consolas,monospace;background:#0b0b0b;color:#cfd0d1;padding:12px;border-radius:6px}"
            "</style>"
        )
        html = f"{styles}<pre>{code_escaped}</pre>"
        return html

    # -----------------------------
    # Internal parsing functions (adapted)
    # -----------------------------
    def _fast_extract_modules_optimized(self, code: str) -> List[Dict[str, str]]:
        modules = []
        lines = code.splitlines()
        n = len(lines)
        MAX_LINES = 200000
        if n > MAX_LINES:
            lines = lines[:MAX_LINES]
            n = MAX_LINES
        inside = False
        buf = []
        name = ""
        module_count = 0
        MAX_MODULES = 500
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not inside and stripped.startswith("module"):
                inside = True
                buf = [line]
                m = re.match(r"module\s+(\w+)", stripped)
                name = m.group(1) if m else "unknown"
                continue
            if inside:
                buf.append(line)
                if stripped.startswith("endmodule"):
                    inside = False
                    modules.append({"name": name, "text": "\n".join(buf)})
                    module_count += 1
                    buf = []
                    name = ""
                    if module_count >= MAX_MODULES:
                        break
        return modules

    def _generate_explanation_optimized(self, code: str, modules: List[Dict[str, str]]) -> str:
        lines = []
        lines.append(f"EXPLANATION GENERATED: {datetime.utcnow().isoformat()}")
        lines.append("=" * 60)
        lines.append("")
        if not modules:
            lines.append("No modules detected.")
            return "\n".join(lines)
        lines.append(f"MODULES DETECTED: {len(modules)}")
        for m in modules:
            lines.append(f"  - {m['name']}")
        lines.append("")
        module_data = []
        for idx, mod in enumerate(modules):
            text = mod["text"]
            insts = re.findall(r'\b(\w+)\s+(\w+)\s*\(', text[:50000])
            header_match = re.search(r'\((.*?)\);', text[:2000], flags=re.DOTALL)
            port_block = header_match.group(1) if header_match else ""
            inputs = len(re.findall(r'\binput\b', port_block))
            outputs = len(re.findall(r'\boutput\b', port_block))
            inouts = len(re.findall(r'\binout\b', port_block))
            module_data.append({"name": mod["name"], "inputs": inputs, "outputs": outputs, "inouts": inouts, "instances": insts[:10], "instance_count": len(insts)})
        if module_data:
            top = max(module_data, key=lambda x: x["instance_count"])
            lines.append(f"TOP MODULE: {top['name']}")
            lines.append(f"Instances inside: {top['instance_count']}")
            lines.append("")
        lines.append("MODULE DETAILS:")
        for m in module_data:
            lines.append("-" * 54)
            lines.append(f"MODULE: {m['name']}")
            lines.append(f"  Inputs     : {m['inputs']}")
            lines.append(f"  Outputs    : {m['outputs']}")
            lines.append(f"  Inouts     : {m['inouts']}")
            lines.append(f"  Instances  : {m['instance_count']}")
            for inst in m["instances"]:
                lines.append(f"      - {inst[0]} {inst[1]}")
            lines.append("")
        # simple metrics
        sample = code[:100000]
        always_count = len(re.findall(r'always\s*@', sample))
        assign_count = len(re.findall(r'\bassign\b', sample))
        total = len(code.splitlines())
        lines.append("DESIGN METRICS:")
        lines.append(f"  Total lines: {total}")
        lines.append(f"  always@* (sampled):   {always_count}")
        lines.append(f"  assign (sampled):     {assign_count}")
        return "\n".join(lines)

    # Reuse the comprehensive builders from earlier adapted versions
    def _extract_detailed_module_info(self, code: str) -> Optional[Dict[str, Any]]:
        cleaned = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace("\r", " ")
        module_pat = re.compile(
            r"module\s+(\w+)\s*"
            r"(#\s*\((.*?)\))?\s*"
            r"\((.*?)\)\s*;",
            re.DOTALL | re.MULTILINE
        )
        m = module_pat.search(cleaned)
        if not m:
            return None
        module_name = m.group(1)
        param_block = m.group(3) or ""
        port_block = m.group(4) or ""
        info = {"module_name": module_name, "parameters": [], "inputs": [], "outputs": [], "inouts": [], "clock": None, "reset": None, "addr_width": None, "data_width": None, "optional_apb": {"slverr": False, "pstrb": False, "pprot": False}}
        # params
        for pm in re.finditer(r"parameter\s+(?:\w+\s+)?(\w+)\s*=\s*([^, )\n]+)", param_block, re.IGNORECASE):
            info["parameters"].append({"name": pm.group(1).strip(), "value": pm.group(2).strip()})
        # ports via parse helper
        info["inputs"] = self._parse_port_list(port_block, "input")
        info["outputs"] = self._parse_port_list(port_block, "output")
        info["inouts"] = self._parse_port_list(port_block, "inout")
        all_ports = info["inputs"] + info["outputs"] + info["inouts"]
        for p in all_ports:
            low = p["name"].lower()
            if any(k in low for k in ["clk", "pclk", "clock", "aclk", "hclk"]) and not info["clock"]:
                info["clock"] = p["name"]
            if any(k in low for k in ["resetn", "presetn", "rst_n", "reset", "rst"]) and not info["reset"]:
                info["reset"] = p["name"]
        # widths from parameters
        for prm in info["parameters"]:
            ln = prm["name"].lower()
            try:
                val = int(eval(prm["value"]))
            except Exception:
                val = None
            if val is None:
                continue
            if "addr" in ln:
                info["addr_width"] = val
            if "data" in ln:
                info["data_width"] = val
        low = cleaned.lower()
        info["optional_apb"]["slverr"] = bool(re.search(r"\bpslverr\b", low))
        info["optional_apb"]["pstrb"] = bool(re.search(r"\bpstrb\b", low))
        info["optional_apb"]["pprot"] = bool(re.search(r"\bpprot\b", low))
        info["is_sequential"] = bool(re.search(r"always\s*@\s*\(.*posedge", cleaned))
        info["has_fsm"] = bool(re.search(r"\b(case|unique\s+case)\b", cleaned))
        return info

    def _parse_port_list(self, ports_str: str, direction: str) -> List[Dict[str, Any]]:
        text = ports_str or ""
        text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        parts, cur, depth = [], "", 0
        for ch in text:
            if ch in "[{(":
                depth += 1
            elif ch in "]})":
                depth -= 1 if depth > 0 else 0
            if ch == "," and depth == 0:
                if cur.strip():
                    parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur.strip())
        results = []
        dir_re = r"\b(input|output|inout)\b"
        last_prefix = None
        name_only_pat = re.compile(r"^\s*(\w+)\s*(\[[^\]]+\])?\s*$", flags=re.IGNORECASE)
        for raw in parts:
            piece = raw.strip()
            has_direction = re.search(dir_re, piece) is not None
            if has_direction:
                last_prefix = piece
                explicit_dir = re.search(dir_re, piece, flags=re.IGNORECASE).group(1).lower()
                if explicit_dir != direction:
                    continue
                to_parse = piece
            else:
                if not last_prefix:
                    continue
                if not last_prefix.lower().startswith(direction):
                    continue
                to_parse = last_prefix + " " + piece
            mo = name_only_pat.search(to_parse)
            if mo:
                name = mo.group(1)
                rngs = re.findall(r"\[[^\]]+\]", to_parse)
                packed = rngs[-1] if rngs else ""
                w = self._width_from_range(packed)
                results.append({"name": name, "width": w, "is_bus": w > 1, "range": packed})
        return results

    def _width_from_range(self, rng: str) -> int:
        m = re.match(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", rng or "")
        if not m:
            return 1
        msb, lsb = int(m.group(1)), int(m.group(2))
        return abs(msb - lsb) + 1

    # simplified builders
    def _build_uvm_testbench(self, info: Dict[str, Any]) -> str:
        mod = info.get('module_name', 'top')
        params = info.get('parameters', [])
        clk = info.get('clock') or 'clk'
        rst = info.get('reset') or 'rst_n'
        inputs = info.get('inputs', [])
        outputs = info.get('outputs', [])
        inouts = info.get('inouts', [])
        addr_width = info.get('addr_width') or 32
        data_width = info.get('data_width') or 32
        tb = []
        tb.append(f"// Auto-generated UVM-like TB for {mod}")
        tb.append("`timescale 1ns/1ps")
        tb.append("interface tb_if;")
        tb.append(f"  parameter int ADDR_W = {addr_width};")
        tb.append(f"  parameter int DATA_W = {data_width};")
        tb.append(f"  logic {clk};")
        tb.append(f"  logic {rst};")
        ban = {clk, rst}
        for p in inputs:
            if p['name'] not in ban:
                tb.append(f"  logic {p.get('range','')} {p['name']};")
        for p in outputs:
            if p['name'] not in ban:
                tb.append(f"  logic {p.get('range','')} {p['name']};")
        tb.append("endinterface: tb_if")
        tb.append("\nmodule tb_top;")
        tb.append("  tb_if tb_vif();")
        tb.append(f"  always #5 tb_vif.{clk} = ~tb_vif.{clk};")
        tb.append("  initial begin")
        tb.append(f"    tb_vif.{clk} = 0;")
        tb.append(f"    tb_vif.{rst} = 0;")
        tb.append("    #20;")
        tb.append(f"    tb_vif.{rst} = 1;")
        tb.append("  end")
        param_map = ""
        if params:
            param_map = "#(\n"
            for p in params:
                param_map += f"    .{p['name']}({p['name']}),\n"
            param_map = param_map.rstrip(",\n") + "\n  ) "
        seen = {clk, rst}
        port_lines = [f".{clk}(tb_vif.{clk})", f".{rst}(tb_vif.{rst})"]
        for p in inputs + outputs + inouts:
            if p['name'] not in seen:
                port_lines.append(f".{p['name']}(tb_vif.{p['name']})")
                seen.add(p['name'])
        port_map = ",\n    ".join(port_lines)
        tb.append(f"  {mod} {param_map}uut (")
        tb.append(f"    {port_map}")
        tb.append("  );")
        tb.append("endmodule: tb_top")
        return "\n".join(tb)

    def _build_comprehensive_testbench(self, info: Dict[str, Any]) -> str:
        return self._build_uvm_testbench(info)

    def _build_apb_testbench(self, info: Dict[str, Any], apb_signals: Dict[str, str]) -> str:
        mod = info.get("module_name", "top_apb")
        lines = [f"// APB-style TB skeleton for {mod}", f"// Detected signals: {', '.join(apb_signals.keys()) if apb_signals else 'none'}"]
        lines.append(self._build_uvm_testbench(info))
        return "\n".join(lines)

    def _detect_apb_protocol(self, sample: str) -> Tuple[bool, Dict[str, str]]:
        low = (sample or "").lower()
        want = ["paddr", "psel", "pwrite", "pwdata", "prdata"]
        signals = {}
        found = False
        for w in want:
            if re.search(rf"\b{w}\b", low):
                signals[w] = w
                found = True
        return found, signals

