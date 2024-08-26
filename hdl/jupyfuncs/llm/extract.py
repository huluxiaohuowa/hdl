import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import io
from spire.doc import Document
from spire.doc.common import *
# from ..path.glob import (
#     get_current_dir,
#     get_files
# )


class DocExtractor():
    def __init__(
        self,
        doc_files: list,
        lang: str = "chi_sim"
    ) -> None:
        self.doc_files = doc_files
        self.lang = lang

    @classmethod
    def text_from_doc(
        doc_path
    ):
        document = Document()
        # Load a Word document
        document.LoadFromFile(doc_path)
        document_text = document.GetText()
        return document_text

    @staticmethod
    def text_from_plain(
        txt_path
    ):
        with open(txt_path, "r") as f:
            text = f.read()
        return text

    @staticmethod
    def extract_text_from_image(
        image: Image.Image,
    ) -> str:
        return pytesseract.image_to_string(image, lang=self.lang)

    @staticmethod
    def is_within_bbox(
        bbox1, bbox2
    ):
        """Check if bbox1 is within bbox2."""
        return bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]

    def text_tables_from_pdf(
        self,
        pdf_path,
        table_from_pic: bool = False
    ):
        all_tables = []
        all_texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                tables = page.find_tables()
                page_text = page.extract_text(x_tolerance=0.1, y_tolerance=0.1) or ''
                page_text_lines = page_text.split('\n')

                # Extract tables
                if tables:
                    for table in tables:
                        if table and len(table.extract()) > 1:
                            table_data = table.extract()
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            df['Page'] = page_number + 1  # 添加页码信息
                            all_tables.append(df)

                # Get bounding boxes for tables
                table_bboxes = [table.bbox for table in tables]

                # Filter out text within table bounding boxes
                non_table_text = []
                for char in page.chars:
                    char_bbox = (char['x0'], char['top'], char['x1'], char['bottom'])
                    if not any(self.is_within_bbox(char_bbox, table_bbox) for table_bbox in table_bboxes):
                        non_table_text.append(char['text'])
                remaining_text = ''.join(non_table_text).strip()
                if remaining_text:
                    all_texts.append(remaining_text)

                # Extract tables from images if specified
                if table_from_pic:
                    for img in page.images:
                        try:
                            x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                            if x0 < 0 or top < 0 or x1 > page.width or bottom > page.height:
                                print(f"Skipping image with invalid bounds on page {page_number + 1}")
                                continue

                            cropped_image = page.within_bbox((x0, top, x1, bottom)).to_image()
                            img_bytes = io.BytesIO()
                            cropped_image.save(img_bytes, format='PNG')
                            img_bytes.seek(0)
                            pil_image = Image.open(img_bytes)

                            ocr_text = self.extract_text_from_image(pil_image, lang=self.lang)

                            table = [line.split() for line in ocr_text.split('\n') if line.strip()]

                            if table:
                                num_columns = max(len(row) for row in table)
                                for row in table:
                                    if len(row) != num_columns:
                                        row.extend([''] * (num_columns - len(row)))

                                df = pd.DataFrame(table[1:], columns=table[0])
                                df['Page'] = page_number + 1
                                all_tables.append(df)
                        except Exception as e:
                            print(f"Error processing image on page {page_number + 1}: {e}")

        if all_tables:
            return all_texts, all_tables
        else:
            return all_texts, [pd.DataFrame()]
