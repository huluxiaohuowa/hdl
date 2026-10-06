# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/llm/extract.py
# 说明：大模型接口工具
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
        """Notebook 侧的文档抽取器初始化：只把待处理文档的路径列表 doc_files 与语言 lang（OCR/文本默认简体中文 chi_sim）挂到实例上，
        构造阶段不做任何读文件、加载模型等 IO，具体正文由同类的 text_from_doc / text_from_plain / text_tables_from_pdf 按需取。"""
        self.doc_files = doc_files
        self.lang = lang

    @classmethod
    def text_from_doc(
        doc_path
    ):
        """在 Notebook 里读单个 Word（.doc/.docx）文档：用 spire.doc 的 Document.LoadFromFile 打开 doc_path，GetText() 取回整篇正文纯文本，
        表格与图片不保留，也不写出任何文件。注意声明成 classmethod 后首个形参绑定的是类本身，实际调用需显式补上文档路径。"""
        document = Document()
        # Load a Word document
        document.LoadFromFile(doc_path)
        document_text = document.GetText()
        return document_text

    @staticmethod
    def text_from_plain(
        txt_path
    ):
        """按 open 默认编码把 txt_path 指向的纯文本文件整篇读成字符串返回，便于在 Notebook 单元格里直接查看或继续切分；
        不指定 encoding、也不做换行清洗，中文文件在 locale 非 UTF-8 的环境下会按本地编码解码。"""
        with open(txt_path, "r") as f:
            text = f.read()
        return text

    @staticmethod
    def extract_text_from_image(
        image: Image.Image,
    ) -> str:
        """用 pytesseract 对传入的 PIL 图像做 OCR（光学字符识别），返回识别出的文本字符串，识别语言取 self.lang（默认 chi_sim 简体中文）。
        当前声明为 @staticmethod 却在函数体内引用 self.lang，且签名里没有 lang 形参，
        与下面 text_tables_from_pdf 中 self.extract_text_from_image(pil_image, lang=self.lang) 的调用方式不一致。"""
        return pytesseract.image_to_string(image, lang=self.lang)

    @staticmethod
    def is_within_bbox(
        bbox1, bbox2
    ):
        """判断 bbox1 是否被 bbox2 完全包住：按 [x_min, y_min, x_max, y_max] 四条边界逐一比较，全部满足才为 True，
        不涉及面积或中心点；下面的 PDF 抽取用它把落在表格 bbox 内的字符从正文里剔除。
        Check if bbox1 is within bbox2."""
        return bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3]

    def text_tables_from_pdf(
        self,
        pdf_path,
        table_from_pic: bool = False
    ):
        """用 pdfplumber 打开 pdf_path 逐页抽取，返回 (正文列表, 表格 DataFrame 列表)，供 Notebook 里的入库前预处理：
        表格来自 page.find_tables，首行当表头转 DataFrame 并追加 Page 列（页码从 1 开始）；正文由 page.chars 的
        (x0, top, x1, bottom) 经 is_within_bbox 剔掉落在表格框内的字符后拼接而成；table_from_pic=True 时再对页内每张图
        用 within_bbox 裁图转 PNG 走 OCR 补成表格（坐标越界的图跳过，单图异常只打印），全篇无表格则第二项返回 [空 DataFrame]。"""
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
