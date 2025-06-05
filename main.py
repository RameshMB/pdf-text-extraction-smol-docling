# Prerequisites:
# pip install torch torchvision
# pip install transformers
# pip install opencv-python pillow pytesseract

import torch
from PIL import Image, ImageEnhance
import pytesseract
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import json
import re
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_image(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    resized = enhanced.resize((1024, 1024)).convert("RGB")
    return resized

def get_prompt():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "You are a document extraction model. "
                        "Output structured layout tags in DocTags format. "
                        "Do NOT return markdown or natural language. "
                        "Wrap everything in <document>...</document>. "
                        "Focus on invoice_number, date, vendor, item_table, subtotal, tax, total etc."
                    )
                },
            ]
        }
    ]

def parse_doctags_to_json(doctag_text: str):
    json_obj = {}
    rows = []
    for line in doctag_text.split("<nl>"):
        cells = re.findall(r"<fcel>([^<]+)", line)
        if cells:
            rows.append(cells)
    if rows:
        json_obj["table"] = rows
    return json_obj

def main(image_path: str):
    raw_image = load_image(image_path)
    image = enhance_image(raw_image)

    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    messages = get_prompt()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    prompt_len = inputs.input_ids.shape[1]
    decoded_output = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=False)[0].strip()

    # Wrap with <document> tag if not present
    if "<document>" not in decoded_output:
        decoded_output = "<document>\n" + decoded_output
    if "</document>" not in decoded_output:
        decoded_output += "\n</document>"

    with open("docling_output.doctags.txt", "w", encoding="utf-8") as f:
        f.write(decoded_output)
    print(" Saved raw doctags to: docling_output.doctags.txt")

    try:
        parsed_json = parse_doctags_to_json(decoded_output)
        if parsed_json:
            with open("invoice_output.json", "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2)
            print(" Structured JSON saved to: invoice_output.json")
        else:
            raise ValueError("No table found in doctags.")
    except Exception as e:
        print(f" Docling parse failed: {e}")
        fallback_path = "fallback_cleaned_output.txt"
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(decoded_output)
        print(f" Fallback doctags saved to: {fallback_path}")

        print("\n SmolDocling failed. Falling back to OCR...")
        ocr_text = pytesseract.image_to_string(image)
        with open("fallback_ocr_output.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(" OCR fallback saved to: fallback_ocr_output.txt")

if __name__ == "__main__":
    invoice_image_path = "./image/invoice_sample5.jpg"
    if not os.path.exists(invoice_image_path):
        print(f" File not found: {invoice_image_path}")
    else:
        main(invoice_image_path)


