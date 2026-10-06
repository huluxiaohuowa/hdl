# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/chatgr.py
# 说明：大模型调用封装
# 模块功能：把 OpenAI_M 的流式输出（stream）接到 Gradio 网页聊天界面上，作为可直接启动的演示脚本。
import argparse
import gradio as gr
from .chat import OpenAI_M

# 定义流式输出的生成函数
def chat_with_llm(user_input, chat_history=[]):
    """
    Gradio 聊天回调生成器：先把用户消息与空的 Bot 回复占位追加进历史，再逐块累加 llm.stream 的分块文本并反复 yield 刷新界面。
    Yields: (清空输入框, 聊天历史, 聊天历史)；模块级 llm 由 __main__ 分支创建。

    Generates a response from the LLM based on the given user input and chat history.
    Args:
        user_input (str): The user input message.
        chat_history (list): A list of tuples representing the chat history, where each tuple contains the user's message and the bot's response.
    Yields:
        tuple: A tuple containing the bot's response, the updated chat history, and the original chat history.
    """

    chat_history.append(("User: " + user_input, "Bot: "))  # 初始先追加用户消息
    yield "", chat_history, chat_history  # 返回用户消息

    bot_message = ""  # Bot 消息初始化为空
    resp = llm.stream(
        "你的身份是VIVIBIT人工智能小助手，由芯途异构公司(ICTrek)研发，请回答如下问题，并保证回答所采用的语言与用户问题的语言保持一致。\n"
        "Your identity is VIVIBIT AI Assistant, developed by ICTrek. Please answer the following question and ensure that the language used in the response matches the language of the user’s question.\n Question: "
        + user_input
    )  # 获取流式响应

    for chunk in resp:
        bot_message += chunk  # 累加流式输出
        chat_history[-1] = ("User: " + user_input, "Bot: " + bot_message)
        yield "", chat_history, chat_history  # 每次输出更新后的聊天记录

# 构建 Gradio 界面
def create_demo():
    """
    搭 Gradio 界面：gr.State 存聊天历史，Chatbot 显示历史，输入框的回车与发送按钮都绑到 chat_with_llm 生成器（queue=True 保证逐块刷新）。
    Returns: gr.Blocks 界面对象，由调用方 launch 起服务。

    Creates a Gradio demo interface for a chatbot application.
    The interface includes:
    - A chat history display at the top of the page.
    - A user input textbox at the bottom of the page.
    - A send button to submit messages.
    The user can send messages either by clicking the send button or by pressing the Enter key.
    Returns:
        gr.Blocks: The Gradio Blocks object representing the demo interface.
    """

    with gr.Blocks() as demo:
        chat_history = gr.State([])  # 存储聊天历史
        output = gr.Chatbot(label="Chat History")  # 聊天记录在页面顶端

        with gr.Row():  # 用户输入框在页面底端
            chatbox = gr.Textbox(
                label="Your Message", placeholder="Type your message here...", show_label=False
            )
            send_button = gr.Button("Send")

        # 绑定发送消息的交互
        send_button.click(chat_with_llm, [chatbox, chat_history], [chatbox, output, chat_history], queue=True)
        chatbox.submit(chat_with_llm, [chatbox, chat_history], [chatbox, output, chat_history], queue=True)  # 支持回车发送

    return demo

if __name__ == "__main__":
    # 使用 argparse 处理命令行参数
    parser = argparse.ArgumentParser(description="Gradio LLM Chatbot")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="The host to launch the app on.")  # 改为 -H
    parser.add_argument("-P", "--port", type=int, default=10077, help="The port to launch the app on.")
    parser.add_argument("--llm-host", type=str, default="127.0.0.1", help="The LLM server IP.")
    parser.add_argument("--llm-port", type=int, default=22277, help="The LLM server port.")
    args = parser.parse_args()

    args_dict = {
        key.replace("-", "_"): value
        for key, value in vars(args).items()
    }

    # 初始化连接到 LLM 服务器的接口，使用传入的 host 和 port
    # 赋成模块级全局 llm，供 chat_with_llm 内部按名字直接引用
    llm = OpenAI_M(
        server_ip=args_dict["llm_host"],
        server_port=args_dict["llm_port"]
    )

    # 启动 Gradio 应用
    demo = create_demo()
    demo.launch(server_name=args.host, server_port=args.port)


