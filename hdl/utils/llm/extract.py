# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/extract.py
# 说明：大模型调用封装
# 模块功能：文档文本抽取（DocExtractor），把 Word/纯文本/PDF/图片里的正文与表格读成字符串和 DataFrame，供检索增强生成（RAG）入库前的预处理。
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import io
from spire.doc import Document
from spire.doc.common import *


class DocExtractor():
    """文档正文抽取器：按格式提供 Word/纯文本/PDF/图片的文本与表格读取，另有 LTP 中文分句能力（给了 ltp_model_path 才加载）。
    关键属性：ltp_model_path（分句模型目录）、lang（OCR/分句语言，默认简体中文 chi_sim）、split（分句函数，未配模型时为 None）。
    典型用法：DocExtractor(lang="chi_sim").text_tables_from_pdf(path) 取回 (正文列表, 表格 DataFrame 列表)。"""
    def __init__(
        self,
        ltp_model_path: str = None,
        lang: str = "chi_sim"
    ) -> None:
        """只记录 ltp_model_path 与语言 lang（默认简体中文 chi_sim），self.split 先置 None；
        仅当传入 ltp_model_path 时才延迟导入 ltp、加载分句模型并把 self.split 设为 StnSplit().split（按标点切句，可直接 self.split(text) 调用）。
        LTP 实例只存在局部变量 ltp 上、未挂到 self，因此模型不会从实例访问；serverless 场景下不传模型目录即完全不依赖 ltp。

        Initialize the object with the specified LTP model path and language.
        
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
        """用 spire.doc 的 Document 从 doc_path 加载 Word（.doc/.docx）文档，GetText() 返回整篇正文纯文本，
        表格与图片不保留，也不写任何文件。注意声明为 classmethod 后首个形参会绑定成类本身，调用时需要显式补上文档路径。"""
        document = Document()
        # Load a Word document
        document.LoadFromFile(doc_path)
        document_text = document.GetText()
        return document_text
    
    @staticmethod
    def text_from_plain(
        txt_path
    ):
        """以 open 默认编码读 txt_path 指向的纯文本文件，整篇一次性 read 成正文字符串返回，不做换行清洗或分句。

        Reads and returns the text content from a plain text file.
        
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
        """用 pytesseract 对 PIL 图像做 OCR 并返回识别出的文本字符串，识别语言取实例的 self.lang（默认 chi_sim 简体中文）。
        当前声明为 @staticmethod 却在函数体内引用 self.lang，且签名里没有 lang 形参，与 text_tables_from_pdf 里
        self.extract_text_from_image(pil_image, lang=self.lang) 的调用方式并不匹配。

        Extracts text from the given image using pytesseract.
        
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
        """按 pdfplumber 的页面坐标约定判断 bbox1 是否完全落在 bbox2 内：只逐个比较四条边界
        [x_min, y_min, x_max, y_max]（即 x0/top/x1/bottom），全部满足才 True，不涉及面积或中心点；
        text_tables_from_pdf 用它把落在表格框内的字符从正文里剔除。

        Check if bbox1 is within bbox2.
        
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
        """用 pdfplumber 逐页读 pdf_path，返回 (正文列表, 表格 DataFrame 列表)：表格由 find_tables 得到，首行作表头转 DataFrame 并加 Page 列（页码从 1 起）；
        正文按 page.chars 逐字符取 (x0, top, x1, bottom) 作为 bbox，用 is_within_bbox 剔除落在表格框内的字符后拼接成一页文本（extract_text 只按 0.1 容差取整页文本，不进入返回值）。
        table_from_pic=True 时再把页内每张图的 bbox 用 within_bbox 裁出、转 PNG 走 OCR 补成表格（越界图跳过，单图异常只打印）；全篇无表格时第二项返回 [空 DataFrame]。

        Extract text and tables from a PDF file.
        
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

    