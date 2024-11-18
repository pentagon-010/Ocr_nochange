import fitz  # PyMuPDF
from docx import Document

# Open the PDF
pdf_path = "input.pdf"
docx_path = "output.docx"
doc = Document()
pdf = fitz.open(pdf_path)

# Iterate through PDF pages
for page_number in range(len(pdf)):
    page = pdf[page_number]
    blocks = page.get_text("dict")["blocks"]
    
    for block in blocks:
        for line in block["lines"]:
            line_text = " ".join([span["text"] for span in line["spans"]])
            if line_text.strip():  # Avoid empty lines
                # Add text to the Word document
                doc.add_paragraph(line_text)

    # Add page break
    if page_number < len(pdf) - 1:
        doc.add_page_break()

# Save to .docx
doc.save(docx_path)
print(f"Saved output to {docx_path}")
