from fpdf import FPDF
import io
from typing import List
import pypdfium2
import streamlit as st
from PIL import Image
from surya.ocr import run_ocr
from surya.languages import CODE_TO_LANGUAGE
from surya.input.langs import replace_lang_with_code
from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.processor import load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings


@st.cache_resource()
def load_det_cached():
    return load_det_model(), load_det_processor()


@st.cache_resource()
def load_rec_cached():
    return load_rec_model(), load_rec_processor()


@st.cache_data()
def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    return png.convert("RGB")


# Function for OCR
def ocr(img, highres_img, langs: List[str]):
    replace_lang_with_code(langs)
    pred = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor, highres_images=[highres_img])[0]

    bboxes = [l.bbox for l in pred.text_lines]
    text = [l.text for l in pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    return rec_img, pred


# Function to create a new PDF with text
def create_pdf_with_text(image_path, bboxes, text_lines, output_path):
    img = Image.open(image_path)
    width, height = img.size

    pdf = FPDF(unit="pt", format=[width, height])
    pdf.add_page()
    pdf.image(image_path, x=0, y=0, w=width, h=height)

    pdf.set_auto_page_break(auto=False)
    pdf.set_font("Arial", size=10)

    for bbox, text in zip(bboxes, text_lines):
        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = bbox
        pdf.set_xy(x_min, y_min)
        pdf.cell(w=x_max - x_min, h=y_max - y_min, txt=text, border=0, ln=0)

    pdf.output(output_path)


# Streamlit Interface
st.title("OCR Extraction and PDF Text Placement Demo")

# Load models
det_model, det_processor = load_det_cached()
rec_model, rec_processor = load_rec_cached()

# File upload
in_file = st.sidebar.file_uploader("Upload PDF or Image:", type=["pdf", "png", "jpg", "jpeg", "webp"])
languages = st.sidebar.multiselect("Languages", sorted(list(CODE_TO_LANGUAGE.values())), default=[])

if in_file:
    filetype = in_file.type
    if "pdf" in filetype:
        page_count = len(open_pdf(in_file))
        page_number = st.sidebar.number_input(f"Page number (1-{page_count}):", min_value=1, value=1)
        pil_image = get_page_image(in_file, page_number, settings.IMAGE_DPI)
        pil_image_highres = get_page_image(in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
    else:
        pil_image = Image.open(in_file).convert("RGB")
        pil_image_highres = pil_image

    # Run OCR
    if st.button("Run OCR"):
        rec_img, pred = ocr(pil_image, pil_image_highres, languages)
        st.image(rec_img, caption="OCR Result", use_column_width=True)

        # Display OCR results
        st.subheader("Extracted Text")
        st.text("\n".join([line.text for line in pred.text_lines]))

        st.subheader("JSON Output")
        st.json(pred.model_dump(), expanded=True)

        # Save the extracted text in a PDF with bounding box positions
        output_path = "output_with_text.pdf"
        with st.spinner("Generating PDF with OCR text..."):
            create_pdf_with_text(
                image_path=pil_image,  # Input image
                bboxes=[line.bbox for line in pred.text_lines],  # Bounding boxes
                text_lines=[line.text for line in pred.text_lines],  # Extracted text
                output_path=output_path,  # Output file
            )
        st.success(f"PDF saved at {output_path}")
        st.download_button(
            label="Download PDF with Text",
            data=open(output_path, "rb").read(),
            file_name="output_with_text.pdf",
            mime="application/pdf",
        )
