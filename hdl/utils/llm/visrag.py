# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/visrag.py
# 说明：大模型调用封装
import argparse
from PIL import Image
import hashlib
import torch
import fitz
import gradio as gr
import os
import numpy as np
import json
# import base64
# import io
from transformers import AutoModel, AutoTokenizer

from .chat import OpenAI_M
from .vis import pilimg_to_base64

def get_image_md5(img: Image.Image):
    """
    计算给定图像的MD5哈希值。

    该函数接收一个PIL.Image对象作为输入，将其转换为字节流，并计算该字节流的MD5哈希值。
    这主要用于在不保存图像的情况下，快速识别或验证图像的内容。

    Args:
        img (Image.Image): 输入的图像，为PIL.Image对象。

    Returns:
        str: 图像的MD5哈希值的十六进制表示字符串。
    """
    # 将图像转换为字节流，以便进行哈希计算
    img_byte_array = img.tobytes()

    # 创建一个MD5哈希对象
    hash_md5 = hashlib.md5()

    # 使用图像的字节流更新哈希对象
    hash_md5.update(img_byte_array)

    # 获取哈希值的十六进制表示字符串
    hex_digest = hash_md5.hexdigest()

    # 返回计算出的MD5哈希值
    return hex_digest

def calculate_md5_from_binary(binary_data):
    """ 计算二进制数据的MD5哈希值。
    参数：
    binary_data (bytes): 二进制数据
    返回值：计算出的MD5哈希值的十六进制表示
    """
    # 初始化MD5哈希对象
    hash_md5 = hashlib.md5()
    # 更新哈希对象以计算二进制数据的MD5
    hash_md5.update(binary_data)
    # 返回计算出的MD5哈希值的十六进制表示
    return hash_md5.hexdigest()

def add_pdf_gradio(pdf_file_binary, progress=gr.Progress(), cache_dir=None, model=None, tokenizer=None):
    """Gradio「上传 PDF」按钮的后端：把整份 PDF 按页做视觉 embedding 入库。
    以 PDF 二进制的 MD5 作为 knowledge_base_name，在 cache_dir/<知识库ID>/ 下写入 src.pdf；
    用 fitz 以 dpi=200 把每一页渲染成 RGB 图，逐页在 torch.no_grad 下调视觉 embedding 模型
    model(text=[''], image=[页图], tokenizer=...) 取 reps 存成 reps.npy，页图按图像 MD5 存为 <md5>.png，MD5 顺序列表存 md5s.txt。
    返回知识库 ID；进度经 progress.tqdm 回显到 Gradio 界面。"""
    model.eval()

    knowledge_base_name = calculate_md5_from_binary(pdf_file_binary)

    this_cache_dir = os.path.join(cache_dir, knowledge_base_name)
    os.makedirs(this_cache_dir, exist_ok=True)

    with open(os.path.join(this_cache_dir, f"src.pdf"), 'wb') as file:
        file.write(pdf_file_binary)

    dpi = 200
    doc = fitz.open("pdf", pdf_file_binary)

    reps_list = []
    images = []
    image_md5s = []

    for page in progress.tqdm(doc):
        pix = page.get_pixmap(dpi=dpi)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_md5 = get_image_md5(image)
        image_md5s.append(image_md5)
        with torch.no_grad():
            reps = model(text=[''], image=[image], tokenizer=tokenizer).reps
        reps_list.append(reps.squeeze(0).cpu().numpy())
        images.append(image)

    for idx in range(len(images)):
        image = images[idx]
        image_md5 = image_md5s[idx]
        cache_image_path = os.path.join(this_cache_dir, f"{image_md5}.png")
        image.save(cache_image_path)

    np.save(os.path.join(this_cache_dir, f"reps.npy"), reps_list)

    with open(os.path.join(this_cache_dir, f"md5s.txt"), 'w') as f:
        for item in image_md5s:
            f.write(item+'\n')

    return knowledge_base_name

def retrieve_gradio(knowledge_base, query, topk, cache_dir=None, model=None, tokenizer=None):
    """按 query 从已入库的知识库里检索 topk 个 PDF 页面图（视觉 RAG 的召回端）。
    读取 cache_dir/<knowledge_base>/md5s.txt 与 reps.npy，把加了检索指令前缀的 query 用 embedding 模型编码，
    与各页面向量做点积（matmul，向量需归一化才等价于余弦相似度）后 torch.topk 取前 topk 页，打开对应 <md5>.png 返回 PIL 图像列表；
    同时把知识库、query 与命中页路径写进 q-<query 的 MD5>.json 供 upvote/downvote 记录偏好。知识库目录不存在时返回 None。"""
    model.eval()

    target_cache_dir = os.path.join(cache_dir, knowledge_base)

    if not os.path.exists(target_cache_dir):
        return None

    md5s = []
    with open(os.path.join(target_cache_dir, f"md5s.txt"), 'r') as f:
        for line in f:
            md5s.append(line.rstrip('\n'))

    doc_reps = np.load(os.path.join(target_cache_dir, f"reps.npy"))

    query_with_instruction = "Represent this query for retrieving relevant document: " + query
    with torch.no_grad():
        query_rep = model(text=[query_with_instruction], image=[None], tokenizer=tokenizer).reps.squeeze(0).cpu()

    query_md5 = hashlib.md5(query.encode()).hexdigest()

    doc_reps_cat = torch.stack([torch.Tensor(i) for i in doc_reps], dim=0)

    similarities = torch.matmul(query_rep, doc_reps_cat.T)

    topk_values, topk_doc_ids = torch.topk(similarities, k=topk)

    images_topk = [Image.open(os.path.join(target_cache_dir, f"{md5s[idx]}.png")) for idx in topk_doc_ids.cpu().numpy()]

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'w') as f:
        f.write(json.dumps(
            {
                "knowledge_base": knowledge_base,
                "query": query,
                "retrieved_docs": [os.path.join(target_cache_dir, f"{md5s[idx]}.png") for idx in topk_doc_ids.cpu().numpy()]
            }, indent=4, ensure_ascii=False
        ))

    return images_topk

# def convert_image_to_base64(image):
#     """Convert a PIL Image to a base64 encoded string."""
#     buffered = io.BytesIO()
#     image.save(buffered, format="PNG")
#     image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return image_base64

def answer_question(images, question, gen_model):
    """用生成端多模态大模型（VLM）基于检索到的页图作答：images 是 Gallery 的返回值，逐项取 image[0] 作为页图路径打开成 RGB，
    按最大宽度、高度累加垂直拼接成一张长图，转成 PNG 的 Base64 data URI 后调 gen_model.chat(prompt=question, images=[长图], stream=False)，
    返回一次性（非流式）的完整回答字符串；页图越多拼接图越长，单次请求的图片体积随之增大。"""
    # Load images from the image paths in images[0]
    pil_images = [Image.open(image[0]).convert('RGB') for image in images]

    # Calculate the total size of the new image (for vertical concatenation)
    widths, heights = zip(*(img.size for img in pil_images))

    # Assuming vertical concatenation, so width is the max width, height is the sum of heights
    total_width = max(widths)
    total_height = sum(heights)

    # Create a new blank image with the total width and height
    new_image = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    y_offset = 0
    for img in pil_images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height  # Move the offset down by the height of the image

    # Optionally save or display the final concatenated image (for debugging)
    # new_image.save('concatenated_image.png')

    # Convert the concatenated image to base64
    new_image_base64 = pilimg_to_base64(new_image)

    # Call the model with the base64-encoded concatenated image
    answer = gen_model.chat(
        prompt=question,
        images=[new_image_base64],  # Use the concatenated image
        stream=False
    )
    return answer

def upvote(knowledge_base, query, cache_dir):
    """记录点赞反馈：按 query 的 MD5 找到 cache_dir/<knowledge_base>/q-<md5>.json（retrieve_gradio 写入的检索记录），
    加上 user_preference="upvote" 后另存为 q-<md5>-withpref.json；原文件不改写，无返回值。"""
    target_cache_dir = os.path.join(cache_dir, knowledge_base)
    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "upvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

def downvote(knowledge_base, query, cache_dir):
    """记录点踩反馈：与 upvote 同一套流程，读取 q-<md5>.json 后把 user_preference 置为 "downvote"，
    写到同目录的 q-<md5>-withpref.json，用作检索/生成两阶段的偏好标注；无返回值。"""
    target_cache_dir = os.path.join(cache_dir, knowledge_base)
    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "downvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniCPMV-RAG-PDFQA Script")
    parser.add_argument('--cache-dir', dest='cache_dir', type=str, required=True, help='Cache directory path')
    parser.add_argument('--device', dest='device', type=str, default='cuda:0', help='Device for model inference')
    parser.add_argument('--model-path', dest='model_path', type=str, required=True, help='Path to the embedding model')
    parser.add_argument('--llm-host', dest='llm_host', type=str, default='127.0.0.1', help='LLM server IP address')
    parser.add_argument('--llm-port', dest='llm_port', type=int, default=22299, help='LLM server port')
    parser.add_argument('--server-name', dest='server_name', type=str, default='0.0.0.0', help='Gradio server name')
    parser.add_argument('--server-port', dest='server_port', type=int, default=10077, help='Gradio server port')

    args = parser.parse_args()

    print("Loading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    print("Embedding model loaded!")

    gen_model = OpenAI_M(
        server_ip=args.llm_host,
        server_port=args.llm_port
    )

    with gr.Blocks() as app:
        gr.Markdown("# RAG-PDFQA: Two Vision Language Models Enable End-to-End RAG")

        file_input = gr.File(type="binary", label="Step 1: Upload PDF")
        file_result = gr.Text(label="Knowledge Base ID")
        process_button = gr.Button("Process PDF")

        process_button.click(lambda pdf: add_pdf_gradio(pdf, cache_dir=args.cache_dir, model=model, tokenizer=tokenizer),
                             inputs=file_input, outputs=file_result)

        kb_id_input = gr.Text(label="Knowledge Base ID")
        query_input = gr.Text(label="Your Question")
        topk_input = gr.Number(value=5, minimum=1, maximum=10, step=1, label="Number of pages to retrieve")
        retrieve_button = gr.Button("Retrieve Pages")
        images_output = gr.Gallery(label="Retrieved Pages")

        retrieve_button.click(lambda kb, query, topk: retrieve_gradio(kb, query, topk, cache_dir=args.cache_dir, model=model, tokenizer=tokenizer),
                              inputs=[kb_id_input, query_input, topk_input], outputs=images_output)

        button = gr.Button("Answer Question")
        gen_model_response = gr.Textbox(label="Answer")

        button.click(lambda images, question: answer_question(images, question, gen_model),
                     inputs=[images_output, query_input], outputs=gen_model_response)

        upvote_button = gr.Button("🤗 Upvote")
        downvote_button = gr.Button("🤣 Downvote")

        upvote_button.click(lambda kb, query: upvote(kb, query, cache_dir=args.cache_dir),
                            inputs=[kb_id_input, query_input], outputs=None)
        downvote_button.click(lambda kb, query: downvote(kb, query, cache_dir=args.cache_dir),
                              inputs=[kb_id_input, query_input], outputs=None)

    app.launch(server_name=args.server_name, server_port=args.server_port)