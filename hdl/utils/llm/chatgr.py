import argparse
import gradio as gr
from .chat import OpenAI_M

# 定义流式输出的生成函数
def chat_with_llm(user_input, chat_history=[]):
    """
    Facilitates a chat interaction with a language model (LLM).
    This function takes user input and maintains a chat history. It streams the response from the LLM and updates the chat history in real-time.
    Args:
        user_input (str): The input message from the user.
        chat_history (list, optional): A list of tuples representing the chat history. Each tuple contains two strings: the user's message and the bot's response. Defaults to an empty list.
    Yields:
        tuple: A tuple containing three elements:
            - An empty string (for compatibility with certain frameworks).
            - The updated chat history including the latest user message and the bot's response.
            - The same updated chat history.
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
    Creates a Gradio demo interface for a chatbot application.
    The interface consists of:
    - A chat history display at the top of the page.
    - A user input textbox and a send button at the bottom of the page.
    The send button and the enter key are both bound to the `chat_with_llm` function,
    which handles sending the user's message and updating the chat history.
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
    llm = OpenAI_M(
        server_ip=args_dict["llm_host"],
        server_port=args_dict["llm_port"]
    )

    # 启动 Gradio 应用
    demo = create_demo()
    demo.launch(server_name=args.host, server_port=args.port)


