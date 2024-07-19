import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import io
from spire.doc import Document
from spire.doc.common import *


class DocExtractor():
    def __init__(
        self,
        ltp_model_path: str = None,
        lang: str = "chi_sim"
    ) -> None:
        """Initialize the object with the specified LTP model path and language.
        
        Args:
            ltp_model_path (str): The file path to the LTP model. Default is None.
            lang (str): The language to be used for processing. Default is "chi_sim".
        
        Returns:
            None
        """
        self.ltp_model_path = ltp_model_path
        self.lang = lang

        self.split = None
        if self.ltp_model_path is not None:
            from ltp import StnSplit, LTP
            ltp  = LTP(self.ltp_model_path)
            self.split = StnSplit().split
            # sents = self.split.split(text)
        

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
        """Reads and returns the text content from a plain text file.
        
            Args:
                txt_path (str): The path to the plain text file.
        
            Returns:
                str: The text content read from the file.
        """
        with open(txt_path, "r") as f:
            text = f.read()
        return text
    
    @staticmethod
    def extract_text_from_image(
        image: Image.Image,
    ) -> str:
        """Extracts text from the given image using pytesseract.
        
        Args:
            image (PIL.Image.Image): The input image from which text needs to be extracted.
        
        Returns:
            str: The extracted text from the image.
        """
        return pytesseract.image_to_string(image, lang=self.lang)

    @staticmethod
    def is_within_bbox(
        bbox1, bbox2
    ):
        """Check if bbox1 is within bbox2.
        
        Args:
            bbox1 (list): List of 4 integers representing the bounding box coordinates [x_min, y_min, x_max, y_max].
            bbox2 (list): List of 4 integers representing the bounding box coordinates [x_min, y_min, x_max, y_max].
        
        Returns:
            bool: True if bbox1 is within bbox2, False otherwise.
        """
        return bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]

    def text_tables_from_pdf(
        self,
        pdf_path,
        table_from_pic: bool = False
    ):
        """Extract text and tables from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file.
            table_from_pic (bool, optional): Whether to extract tables from images in the PDF. Defaults to False.
        
        Returns:
            tuple: A tuple containing a list of extracted texts and a list of extracted tables as DataFrames.
        """
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

    