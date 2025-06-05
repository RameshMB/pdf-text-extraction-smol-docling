 🧾 Invoice Data Extractor using SmolDocling & OCR Fallback

This project extracts structured data from invoice images using the SmolDocling vision-language model. If the model fails, it gracefully falls back to OCR using Tesseract. The output is a clean JSON containing fields like invoice number, vendor, date, items, subtotal, and total.

---

 🚀 Features

Uses `ds4sd/SmolDocling-256M-preview` for layout-aware document understanding.
Preprocesses images using contrast enhancement and resizing.
Falls back to Tesseract OCR in case of model failure.
Outputs structured JSON and intermediate raw tags.
Lightweight, no template dependency.

---

 📁 Project Structure

```

Invoice\_Markdown/
├── image/                     # Folder containing invoice images
├── main.py                   # Main script for processing
├── requirements.txt          # Python dependencies
├── docling\_output.doctags.txt    # Model output (DocTags)
├── invoice\_output.json           # Final structured data
├── fallback\_\*.txt               # OCR fallback outputs
├── smoldocling\_env/           # Conda environment (optional)

````

---

 ⚙️ Setup Instructions

 1. Clone or Create Project Folder
Create a folder manually (e.g., `Invoice_Markdown`) and add your invoice images to a subfolder named `image`.

 2. Set Up Python Environment
Create a virtual environment (preferably using Conda):

Option 1-

```bash
conda create -n smoldocling_env python=3.11 -y
conda activate smoldocling_env
````
Option 2-

# Navigate to your project directory
cd Invoice_Markdown

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate

# For macOS/Linux:
source venv/bin/activate


 3. Install Dependencies

Add the following to your `requirements.txt`:

torch
torchvision
transformers
docling\_core
opencv-python
pillow
pytesseract

Then install all:

```bash
pip install -r requirements.txt
```

 4.Install Tesseract OCR

Install [Tesseract](https://github.com/tesseract-ocr/tesseract) on your system.
For Windows, add the binary path to your script using:

```python
pytesseract.pytesseract.tesseract_cmd = r"<absolute_path_to_tesseract_exe>"
```

---

▶️ How to Run

Make sure your desired invoice image is placed in the `image/` folder.
Update the filename path in `main.py` as needed.

Then run:

```bash
python main.py
```

---

📦 Output Files

 `invoice_output.json` – Structured data output
 `docling_output.doctags.txt` – Raw model tags
 `fallback_cleaned_output.txt` – If parsing fails
 `fallback_ocr_output.txt` – OCR fallback text


