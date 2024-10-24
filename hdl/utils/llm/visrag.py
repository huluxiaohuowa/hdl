import argparse
from PIL import Image
import hashlib
import torch
import fitz
import gradio as gr
import os
import numpy as np
import json
from transformers import AutoModel, AutoTokenizer
from hdl.utils.llm.chat import OpenAI_M

def get_image_md5(img: Image.Image):
    img_byte_array = img.tobytes()
    hash_md5 = hashlib.md5()
    hash_md5.update(img_byte_array)
    hex_digest = hash_md5.hexdigest()
    return hex_digest

def calculate_md5_from_binary(binary_data):
    hash_md5 = hashlib.md5()
    hash_md5.update(binary_data)
    return hash_md5.hexdigest()

def add_pdf_gradio(pdf_file_binary, progress=gr.Progress(), cache_dir=None, model=None, tokenizer=None):
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

def answer_question(images, question, gen_model):
    images_ = [Image.open(image[0]).convert('RGB') for image in images]
    answer = gen_model.chat(
        prompt=question,
        images=images_,
        stream=False
    )
    return answer

def upvote(knowledge_base, query, cache_dir):
    target_cache_dir = os.path.join(cache_dir, knowledge_base)
    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "upvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

def downvote(knowledge_base, query, cache_dir):
    target_cache_dir = os.path.join(cache_dir, knowledge_base)
    query_md5 = hashlib.md5(query.encode()).hexdigest()

    with open(os.path.join(target_cache_dir, f"q-{query_md5}.json"), 'r') as f:
        data = json.loads(f.read())

    data["user_preference"] = "downvote"

    with open(os.path.join(target_cache_dir, f"q-{query_md5}-withpref.json"), 'w') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAG-PDFQA Script")
    parser.add_argument('--cache_dir', type=str, required=True, help='Cache directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for model inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the embedding model')
    parser.add_argument('--llm_host', type=str, default='127.0.0.0', help='LLM server IP address')
    parser.add_argument('--llm_port', type=int, default=22299, help='LLM server port')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Gradio server name')
    parser.add_argument('--server_port', type=int, default=10077, help='Gradio server port')

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
        gr.Markdown("# Vision Language Models Enable End-to-End RAG")

        file_input = gr.File(type="binary", label="Step 1: Upload PDF")
        file_result = gr.Text(label="Knowledge Base ID")
        process_button = gr.Button("Process PDF")

        process_button.click(add_pdf_gradio, inputs=[file_input], outputs=file_result,
                             _kwargs={'cache_dir': args.cache_dir, 'model': model, 'tokenizer': tokenizer})

        kb_id_input = gr.Text(label="Knowledge Base ID")
        query_input = gr.Text(label="Your Question")
        topk_input = gr.Number(value=5, minimum=1, maximum=10, step=1, label="Number of pages to retrieve")
        retrieve_button = gr.Button("Retrieve Pages")
        images_output = gr.Gallery(label="Retrieved Pages")

        retrieve_button.click(retrieve_gradio, inputs=[kb_id_input, query_input, topk_input], outputs=images_output,
                              _kwargs={'cache_dir': args.cache_dir, 'model': model, 'tokenizer': tokenizer})

        button = gr.Button("Answer Question")
        gen_model_response = gr.Textbox(label="Model's Answer")

        button.click(answer_question, inputs=[images_output, query_input], outputs=gen_model_response, _kwargs={'gen_model': gen_model})

        upvote_button = gr.Button("🤗 Upvote")
        downvote_button = gr.Button("🤣 Downvote")

        upvote_button.click(upvote, inputs=[kb_id_input, query_input], outputs=None, _kwargs={'cache_dir': args.cache_dir})
        downvote_button.click(downvote, inputs=[kb_id_input, query_input], outputs=None, _kwargs={'cache_dir': args.cache_dir})

    app.launch(server_name=args.server_name, server_port=args.server_port)