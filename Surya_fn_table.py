import io
from typing import List
import pypdfium2
import streamlit as st
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, elevenSeventeen
import fitz  # PyMuPDF for font extraction
import pdfplumber  # For table detection

from surya.detection import batch_text_detection
from surya.model.detection.model import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.languages import CODE_TO_LANGUAGE
from surya.ocr import run_ocr
from surya.settings import settings
from surya.postprocessing.text import draw_text_on_image


# Cache models to optimize loading
@st.cache_resource()
def load_det_cached():
    return load_model(), load_processor()


@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()


# Function for OCR
def ocr(img, highres_img, langs: List[str]):
    img_pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor, highres_images=[highres_img])[0]
    bboxes = [l.bbox for l in img_pred.text_lines]
    text_lines = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text_lines, img.size, langs)
    return rec_img, text_lines, bboxes


# Extract fonts and styles using PyMuPDF
def extract_fonts_and_styles(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    doc = fitz.open(stream)
    font_styles = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_styles.append((span["text"], span["font"], span["size"]))
    return font_styles


# Extract tables using pdfplumber
def extract_tables(pdf_file):
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_file.getvalue())) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.append(table)
    return tables


# Save OCR output to PDF with styles and tables
def save_ocr_with_styles(all_bboxes, fonts, tables, pdf_filename: str):
    pdf = canvas.Canvas(pdf_filename, pagesize=elevenSeventeen)
    pdf_width, pdf_height = elevenSeventeen

    for img_size, bboxes, text_lines, font_style in zip(all_bboxes, fonts):
        img_width, img_height = img_size

        for bbox, text in zip(bboxes, text_lines):
            x0, y0, x1, y1 = bbox
            norm_x = x0 / img_width
            norm_y = y0 / img_height
            pdf_x = norm_x * pdf_width
            pdf_y = (1 - norm_y) * pdf_height
            # Use extracted font style
            pdf.setFont(font_style[1], font_style[2])
            pdf.drawString(pdf_x, pdf_y, text)

        pdf.showPage()

    # Add tables to the end of the document
    for table in tables:
        pdf.drawString(100, pdf_height - 100, "Extracted Table:")
        for row in table:
            pdf.drawString(100, pdf_height - 120, ", ".join(row))
            pdf_height -= 20
        pdf.showPage()

    pdf.save()
    return pdf_filename


# Open PDF and extract page images
def get_all_page_images(pdf_file, dpi=settings.IMAGE_DPI):
    stream = io.BytesIO(pdf_file.getvalue())
    doc = pypdfium2.PdfDocument(stream)
    page_count = len(doc)
    images = []

    for page_num in range(page_count):
        renderer = doc.render(
            pypdfium2.PdfBitmap.to_pil,
            page_indices=[page_num],
            scale=dpi / 72,
        )
        image = list(renderer)[0].convert("RGB")
        images.append(image)
    return images


# Streamlit app UI
st.set_page_config(layout="wide")
st.markdown("""
# OCR Demo with Layout and Table Preservation

This app performs OCR for all pages in a PDF and exports results to a PDF file with font styles and tables.

Notes:
- Works best with printed text documents.
- You can upload PDF or image files (PNG, JPG).
""")

in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
languages = st.sidebar.multiselect(
    "Languages", sorted(list(CODE_TO_LANGUAGE.values())),
    default=[], max_selections=4, help="Select known languages in the document to improve OCR accuracy."
)

if in_file is None:
    st.stop()

filetype = in_file.type

# Process all pages of a PDF or single image
if "pdf" in filetype:
    pil_images = get_all_page_images(in_file)
    fonts = extract_fonts_and_styles(in_file)
    tables = extract_tables(in_file)
else:
    pil_images = [Image.open(in_file).convert("RGB")]
    fonts = []
    tables = []

# Load models
det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()

# OCR processing
if st.sidebar.button("Run OCR"):
    all_bboxes = []

    for page_index, pil_image in enumerate(pil_images):
        st.markdown(f"### Page {page_index + 1}")
        pil_image_highres = pil_image
        rec_img, text_lines, bboxes = ocr(pil_image, pil_image_highres, languages)

        # Display OCR output for each page
        st.image(rec_img, caption=f"OCR Result (Page {page_index + 1})", use_column_width=True)
        st.text("\n".join(text_lines))
        all_bboxes.append((pil_image.size, bboxes, text_lines))

    # Save to PDF with styles and tables
    pdf_filename = "ocr_output_with_styles.pdf"
    save_ocr_with_styles(all_bboxes, fonts, tables, pdf_filename)
    with open(pdf_filename, "rb") as pdf_file:
        st.sidebar.download_button(
            label="Download PDF with Styles and Tables",
            data=pdf_file,
            file_name=pdf_filename,
            mime="application/pdf"
        )
