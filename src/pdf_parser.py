# Functions for parsing PDF documents, extracting text, tables, and images.

import os
import fitz # PyMuPDF
import pdfplumber

def parse_pdf(pdf_path, image_output_dir):
    """
    Parses a PDF document, extracting text, tables, and image metadata.
    Combines text with structured table data for better contextualization.
    Handles errors per page and continues processing.
    """
    pages_data = []
    # Create the directory for extracted images if it doesn't exist
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'.")
        return []

    print(f"Starting PDF parsing for {pdf_path}...")
    try:
        with fitz.open(pdf_path) as doc, pdfplumber.open(pdf_path) as pdf_plumber_doc:
            total_pages = len(doc)
            print(f"Parsing PDF: {pdf_path} with {total_pages} pages...")

            for i in range(total_pages):
                page_num = i + 1
                page_content = ""
                tables = []
                image_info = []
                page_error = None

                print(f"Processing page {page_num} of {total_pages}...")

                try:
                    page_fitz = doc.load_page(i)
                    page_plumber = pdf_plumber_doc.pages[i]

                    # 1. Extract text content
                    text = page_fitz.get_text()

                    # 2. Extract tables and format them as Markdown
                    extracted_tables = page_plumber.extract_tables()
                    table_texts = []
                    for table in extracted_tables:
                        if not table: continue
                        clean_header = [h if h is not None else "" for h in table[0]]
                        clean_rows = [[cell if cell is not None else "" for cell in row] for row in table[1:]]

                        table_md = "\n".join([
                            "| " + " | ".join(clean_header) + " |",
                            "|---" * len(clean_header) + "|",
                            *[("| " + " | ".join(r) + " |") for r in clean_rows]
                        ])
                        table_texts.append(f"\n--- Table on Page {page_num} ---\n{table_md}\n-----------------------------\n")

                    combined_text = text + "\n".join(table_texts)
                    page_content = combined_text
                    tables = [t.strip() for t in table_texts]

                    # 3. Extract image metadata
                    for img_index, img in enumerate(page_fitz.get_images(full=True)):
                        xref = img[0]
                        image_data = doc.extract_image(xref)
                        
                        bbox = image_data.get("bbox")
                        bbox_str = str(bbox) if bbox is not None else "N/A"

                        image_filename = f"page_{page_num}_img_{img_index+1}.{image_data.get('ext', 'png')}"
                        image_path = os.path.join(image_output_dir, image_filename)
                        
                        if "image" in image_data:
                            with open(image_path, "wb") as f:
                                f.write(image_data["image"])
                        else:
                            print(f"Warning: Image {img_index+1} on page {page_num} has no image data. Not saving file.")
                            image_path = "N/A"

                        image_info.append({
                            "filename": image_filename,
                            "path": image_path,
                            "bbox": bbox_str,
                            "page_num": page_num
                        })
                    
                    if image_info:
                        image_ref_text = f"\n\n--- Images on Page {page_num} ---\n"
                        for img in image_info:
                            image_ref_text += f"Image '{img['filename']}' found at coordinates {img['bbox']}.\n"
                        image_ref_text += "--------------------------------\n"
                        page_content += image_ref_text

                except Exception as e:
                    page_error = str(e)
                    print(f"Error processing page {page_num}: {e}")
                    if not page_content:
                        page_content = f"Error: Could not extract content from page {page_num}. Error details: {page_error}"

                pages_data.append({
                    "page_content": page_content,
                    "metadata": {
                        "page": page_num,
                        "source": pdf_path,
                        "tables": tables,
                        "images": image_info,
                        "parsing_error": page_error
                    }
                })
        print(f"Finished parsing {total_pages} pages.")
        return pages_data
    except Exception as e:
        print(f"Critical error opening or parsing PDF: {e}")
        return []