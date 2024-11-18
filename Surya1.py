import streamlit as st
from surya_ocr import OCRProcessor

# Streamlit app title
st.title("Surya-OCR: Extract OCR with Layout and Alignment")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Initialize OCR Processor
    ocr_processor = OCRProcessor()

    try:
        # Perform OCR while preserving layout and alignment
        st.info("Performing OCR... This may take a few seconds.")
        result = ocr_processor.extract_with_layout("temp.pdf")

        # Display OCR Result
        st.subheader("Extracted Text with Layout")
        st.text_area("OCR Output", value=result, height=400)

        # Option to download the extracted OCR result
        st.download_button(
            label="Download OCR Result",
            data=result,
            file_name="ocr_result.txt",
            mime="text/plain",
        )

    except Exception as e:
        st.error(f"Error during OCR extraction: {e}")

# Footer
st.caption("Powered by Surya-OCR and Streamlit")
