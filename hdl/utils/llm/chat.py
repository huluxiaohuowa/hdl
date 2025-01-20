import typing as t
import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
import subprocess
from typing import Generator
import re
import yaml

from openai import OpenAI
from PIL import Image

from ..desc.template import FN_TEMPLATE, COT_TEMPLATE, OD_TEMPLATE
from ..desc.func_desc import TOOL_DESC
from .vis import draw_and_plot_boxes_from_json, to_img, to_base64


def parse_fn_markdown(markdown_text, params_key="params"):
    """
    Parses a markdown text to extract function name and parameters.
    Args:
        markdown_text (str): The markdown text containing the function name and parameters.
        params_key (str, optional): The key under which the parameters will be nested in the result dictionary. Defaults to "params".
    Returns:
        dict: A dictionary containing the function name and parameters. The function name is stored with the key "function_name", and the parameters are nested under the specified params_key.
    """

    lines = markdown_text.strip().split("\n")
    result = {}
    params = {}

    for line in lines:
        # 使用正则提取 key 和 value
        match = re.match(r"-\s*(\w+):\s*(.+)", line.strip())
        if match:
            key, value = match.groups()
            if key == "function_name":
                result[key] = value  # 固定的 function_name
            else:
                params[key] = value  # 其他的进入 params（或替代键名）

    # 将提取的参数嵌套到指定的键名下
    result[params_key] = params
    return result

def parse_fn_markdown(markdown_text, params_key="params"):
    # 将 Markdown 文本按行分割并去掉空白
    lines = markdown_text.strip().split("\n")
    result = {}
    params = {}

    for line in lines:
        # 匹配键值对（即 key: value 格式），支持中英文冒号和空白
        match = re.match(r"-\s*(\w+)\s*[:：]\s*(.+)", line.strip())
        if match:
            key, value = match.groups()
            if key == "function_name":
                result[key] = value.strip()  # 固定的 function_name
            else:
                params[key] = value.strip()  # 其他进入 params

    # 如果有 params_key，就嵌套到该键名下
    result[params_key] = params
    return result

def parse_cot_markdown(markdown_text):
    """
    Parse a Markdown text formatted as 'COT' (Title, Tool, Content, Stop Thinking) and extract relevant information.

    Args:
        markdown_text (str): The Markdown text to parse.

    Returns:
        dict: A dictionary containing the parsed information with the following keys:
            - 'title': Title extracted from the Markdown text.
            - 'tool': Tool extracted from the Markdown text.
            - 'content': Content extracted from the Markdown text.
            - 'stop_thinking': Boolean indicating whether 'stop_thinking' is true or false.

    Note:
        - 'stop_thinking' value is considered True only if it is explicitly 'true' (case-insensitive).

    """
    # 提取标题（支持跨行）
    title_match = re.search(r"##\s*(.+?)(?=\n-|\Z)", markdown_text, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""

    # 提取工具
    tool_match = re.search(r"-\s*tool\s*[:：]\s*(.+?)(?=\n-|\Z)", markdown_text, re.DOTALL)
    tool = tool_match.group(1).strip() if tool_match else ""

    # 提取内容（支持跨行）
    content_match = re.search(r"-\s*content\s*[:：]\s*(.+?)(?=\n-|\Z)", markdown_text, re.DOTALL)
    content = content_match.group(1).strip() if content_match else ""

    # 提取停止思考
    stop_thinking_match = re.search(r"-\s*stop_thinking\s*[:：]\s*(.+?)(?=\n-|\Z)", markdown_text, re.DOTALL)
    stop_thinking = stop_thinking_match.group(1).strip().lower() in ["true"] if stop_thinking_match else False

    # 返回解析结果的字典
    return {
        "title": title,
        "tool": tool,
        "content": content,
        "stop_thinking": stop_thinking
    }



def run_tool_with_kwargs(tool, func_kwargs):
    """Run the specified tool with the provided keyword arguments.

    Args:
        tool (callable): The tool to be executed.
        func_kwargs (dict): The keyword arguments to be passed to the tool.

    Returns:
        The result of executing the tool with the provided keyword arguments.
    """
    return tool(**func_kwargs)


class OpenAI_M:
    def __init__(
        self,
        client_conf: dict = None,
        client_conf_dir: str = None,
        load_conf: bool = True,
        tools: list = None,
        tool_desc: dict = None,
        cot_desc: str = None,
        od_desc: str = None,
        *args,
        **kwargs
    ):
        """
        Initialize an instance of the OpenAI_M class with configuration options.

        Args:
            model_path (str): Path to the model. Defaults to "default_model".
            device (str): Device to use, either 'gpu' or 'cpu'. Defaults to 'gpu'.
            generation_kwargs (dict, optional): Additional keyword arguments for generation.
            server_ip (str): IP address of the server. Defaults to "172.28.1.2".
            server_port (int): Port number of the server. Defaults to 8000.
            api_key (str): API key for authentication. Defaults to "dummy_key".
            use_groq (bool): Flag to use Groq client. Defaults to False.
            groq_api_key (str, optional): API key for Groq client.
            tools (list, optional): List of tools to be used.
            tool_desc (dict, optional): Additional tool descriptions.
            cot_desc (str, optional): Chain of Thought description.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.client_conf = {}
        if client_conf is None:
            assert client_conf_dir is not None
            self.client_conf_path = client_conf_dir
            if load_conf:
                self.load_clients()
        else:
            self.client_conf = client_conf

        # self.clients = {}
        for _, conf in self.client_conf.items():
            conf["client"] = OpenAI(
                base_url=conf["host"],
                api_key=conf.get("api_key", "dummy_key"),
                *args,
                **kwargs
            )

        self.tools: list = tools if tools else []
        self.tool_desc: dict = TOOL_DESC
        if tool_desc is not None:
            self.tool_desc = self.tool_desc | tool_desc

        self.tool_descs = [
            self.tool_desc[tool]['desc']
            for tool in self.tools
        ]
        self.tool_descs_verbose = [
            self.tool_desc[tool]['desc']
            + self.tool_desc[tool]['md']
            for tool in self.tools
        ]

        self.tool_info = "\n".join(self.tool_descs)
        self.tool_desc_str = "\n".join(self.tool_descs_verbose)

        self.cot_desc = cot_desc if cot_desc else COT_TEMPLATE
        self.od_desc = od_desc if od_desc else OD_TEMPLATE

    def add_client(
        self,
        client_id: str,
        host: str,
        port: int = None,
        model: str = "default_model",
        api_key: str = "dummy_key",
        **kwargs
    ):
        self.client_conf[client_id] = {}
        if not host.startswith('http') and port:
            host = f"http://{host}:{port}/v1"
        self.client_conf[client_id]['host'] = host
        self.client_conf[client_id]['model'] = model
        self.client_conf[client_id]['client'] = OpenAI(
            base_url=host,
            api_key=api_key,
            **kwargs
        )

    def load_clients(self):
        with open(self.client_conf_path, 'r') as file:
            data = yaml.safe_load(file)

        # 更新 host 字段
        for _, value in data.items():
            host = value.get('host', '')
            port = value.get('port', '')
            if not host.startswith('http') and port:  # 确保有 port 才处理
                value['host'] = f"http://{host}:{port}/v1"
        self.client_conf = data

    def cot(
        self,
        prompt,
        max_step: int = 30,
        steps: list = None,
        **kwargs
    ):
        # 初始化当前信息为空字符串，用于累积后续的思考步骤和用户问题
        current_info = ""
        # 初始化步数为0，用于控制最大思考次数
        n_steps = 0
        if steps is None:
            steps = []

        # 进入思考循环，直到找到答案或达到最大步数
        while True:
            n_steps += 1
            # 检查是否达到最大步数，如果是，则退出循环并返回默认答案
            if n_steps > max_step:
                print("Max step reached!")
                yield n_steps, current_info, steps
                return

            # 调用思考函数，传入当前信息和用户问题，获取下一步思考的结果
            resp = self.invoke(
                "现有的步骤得出来的信息：\n" + current_info + "\n用户问题：" + prompt,
                sys_info=COT_TEMPLATE + self.tool_info,
                assis_info="好的，我将根据用户的问题和信息给出当前需要进行的操作或最终答案",
                **kwargs
            )

            # print(f"第{n_steps}步思考结果：\n{resp}\n\n")

            try:
                # 将思考结果解析为JSON格式，以便后续处理
                step_json = parse_cot_markdown(resp)
                # print(step_json)
                # 将当前思考步骤添加到步骤列表中
                steps.append(step_json)
                # 如果思考步骤中标记为停止思考，则打印所有步骤并返回最终答案

                # 如果思考步骤中包含使用工具的指示，则构造工具提示并调用agent_response方法
                if 'tool' in step_json and step_json['tool'] in self.tools:
                    tool_prompt = step_json["tool"] \
                        + step_json.get("title", "") \
                        + step_json.get("content", "") \
                        + f"用户问题为：{prompt}"

                    tool_resp = self.agent_response(
                        tool_prompt,
                        stream=False,
                        **kwargs
                    )
                    if isinstance(tool_resp, Generator):
                        tool_resp = "".join(tool_resp)
                    # 将工具返回的信息累积到当前信息中
                    current_info += f"\n{tool_resp}"
                else:
                    if step_json.get("stop_thinking", False):
                        # current_info += f"\n{step_json.get("content", "")}"
                        current_info += f"\n{step_json.get('content', '')}"
                        yield n_steps, current_info, steps
                        return
                    # 如果不使用工具，将当前思考步骤的标题累积到当前信息中
                    else:
                        current_info += f"\n{step_json.get('title', '')}"
                        current_info += f"\n{step_json.get('content', '')}"

                if step_json.get("stop_thinking", False):
                    current_info += f"\n{step_json.get('title', '')}"
                    current_info += f"\n{step_json.get('content', '')}"
                    yield n_steps, current_info, steps
                    return
                yield n_steps, current_info, steps
            except Exception as e:
                # 捕获异常并打印，然后继续下一轮思考
                print(e)
                continue

    def get_resp(
        self,
        prompt: str,
        client_id: str = None,
        sys_info: str = None,
        assis_info: str = None,
        images: list = None,
        image_keys: tuple = ("image_url", "url"),
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        model: str=None,
        stream: bool = True,
        **kwargs: t.Any,
    ):
        """Prepare and send a request to the chat model, and return the model's response.

        Args:
            prompt (str): The user's input text.
            sys_info (str, optional): System information, if any. Defaults to None.
            assis_info (str, optional): Assistant information, if any. Defaults to None.
            images (list, optional): List of images to send with the request, if any. Defaults to None.
            image_keys (tuple, optional): Tuple containing the keys for image information. Defaults to ("image_url", "url").
            stop (_type_, optional): List of stop sequences for the model. Defaults to ["USER:", "ASSISTANT:"].
            model (str, optional): The name of the model to use. Defaults to "default_model".
            stream (bool, optional): Whether to use streaming mode for the response. Defaults to True.

        Returns:
            _type_: The response object from the model.
        """
        if not model:
            model = self.client_conf[client_id]["model"]

        # Initialize the content list with at least the user's text input
        content = [
            {"type": "text", "text": prompt},
        ]

        # Adjust the image_keys to be a tuple of length 3 based on its current length
        if isinstance(image_keys, str):
            image_keys = (image_keys,) * 3
        elif len(image_keys) == 2:
            image_keys = (image_keys[0],) + tuple(image_keys)
        elif len(image_keys) == 1:
            image_keys = (image_keys[0],) * 3

        # If images are provided, append them to the content list
        if images:
            if isinstance(images, str):
                images = [images]
            for img in images:
                content.append({
                    "type": image_keys[0],
                    image_keys[1]: {
                        image_keys[2]: img
                    }
                })
        else:
            # If no images are provided, content is simply the prompt text
            content = prompt

        # Initialize the messages list and add system information if provided
        messages = []
        if sys_info:
            messages.append({
                "role": "system",
                "content": sys_info
            })

        # Add the user's input as a message
        messages.append({
            "role": "user",
            "content": content
        })

        # Add assistant information to the messages list if provided
        if assis_info:
            messages.append({
                "role": "assistant",
                "content": assis_info
            })

        # Call the model to generate a response
        response = self.client_conf[client_id]["client"].chat.completions.create(
            messages=messages,
            stream=stream,
            model=model,
            **kwargs
        )

        # Return the model's response
        return response

    def invoke(
        self,
        *args,
        **kwargs
    ):
        """Invoke the function with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The content of the first choice message in the response.
        """
        response = self.get_resp(*args, stream=False, **kwargs)
        return response.choices[0].message.content

    def stream(
        self,
        *args,
        **kwargs
    ):
        """Stream content from the response in chunks.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Yields:
                str: Content in chunks from the response.
        """
        response = self.get_resp(*args, stream=True, **kwargs)
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content


    def chat(self, *args, stream=True, **kwargs):
        """Call either the stream or invoke method based on the value of the stream parameter.

        Args:
            *args: Variable length argument list.
            stream (bool): A flag to determine whether to call the stream method (default is True).
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of calling either the stream or invoke method based on the value of the stream parameter.
        """
        if stream:
            return self.stream(*args, **kwargs)
        else:
            return self.invoke(*args, **kwargs)

    def agent_response(
        self,
        prompt : str,
        stream = True,
        **kwargs: t.Any
    ):
        """'''Generate agent response based on the given prompt.

        Args:
            prompt (str): The prompt for which agent response is generated.
            stream (bool, optional): Flag to determine if the response should be streamed. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The agent response based on the prompt.
        '''
        """
        decision_dict_str = self.get_decision(prompt, **kwargs)
        decision_dict = parse_fn_markdown(decision_dict_str)
        if decision_dict.get("function_name", None) is None:
            return self.chat(prompt=prompt, stream=stream, **kwargs)
        else:
            tool_result = str(self.get_tool_result(decision_dict_str))
            prompt_final = "根据上下文回答最后的用户问题：\n上下文信息：\n"
            prompt_final += tool_result
            # prompt_final += f"\n用户的问题：\n{prompt}"
            if stream:
                return self.stream(
                    prompt=prompt,
                    sys_info=prompt_final,
                    **kwargs
                )
            else:
                return self.invoke(
                    prompt=prompt,
                    sys_info=prompt_final,
                    **kwargs
                )

    def get_decision(
        self,
        prompt: str,
        **kwargs: t.Any,
    ):
        """Get decision based on the given prompt.

        Args:
            prompt (str): The prompt for decision making.
            **kwargs: Additional keyword arguments for decision making.

        Returns:
            str: The decision dictionary string.
        """
        fn_template = kwargs.pop("fn_template", FN_TEMPLATE)
        prompt_final = fn_template
        # for tool in self.tools:
        #     prompt_final += self.tool_desc.get(tool.__name__, "")
        # prompt_final += f"\n用户的问题：\n{prompt}"

        prompt_final += self.tool_desc_str

        decision_dict_str = self.invoke(
            prompt=prompt,
            sys_info=prompt_final,
            **kwargs
        )
        return decision_dict_str

    def get_tool_result(
        self,
        decision_dict_str: str,
    ):
        """Get the result of a tool based on the decision made.

        Args:
            prompt (str): The prompt to make a decision.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the tool.
        """
        decision_dict = parse_fn_markdown(decision_dict_str)
        func_name = decision_dict.get("function_name", None)
        if func_name is None:
            return ""
        else:
            try:
                tool_final = ''
                for tool in self.tools:
                    if tool == func_name:
                        tool_final = tool
                func_kwargs = decision_dict.get("params")
                if tool_final == "object_detect":
                    func_kwargs["llm"] = self
                return getattr(self.tools, tool_final)(**func_kwargs)
            except Exception as e:
                print(e)
                return ""

    async def get_tool_result_async(
        self,
        prompt,
        **kwargs: t.Any
    ):
        """
        Asynchronous version of the get_tool_result function that can run in parallel using multiprocessing.

        Args:
            prompt (str): The prompt to get the decision for.
            **kwargs: Additional keyword arguments to pass to the decision function.

        Returns:
            str: The result from the selected tool based on the decision made.
        """

        decision_dict_str = await asyncio.to_thread(self.get_decision, prompt, **kwargs)
        try:
            decision_dict = parse_fn_markdown(decision_dict_str)
        except Exception as e:
            print(e)
            return ""
        func_name = decision_dict.get("function_name", None)
        if func_name is None:
            return ""
        else:
            try:
                tool_final = ''
                for tool in self.tools:
                    if tool == func_name:
                        tool_final = tool
                func_kwargs = decision_dict.get("params")

                loop = asyncio.get_running_loop()
                with ProcessPoolExecutor() as pool:
                    result = await loop.run_in_executor(pool, run_tool_with_kwargs, getattr(tools, tool_final), func_kwargs)
                return result
            except Exception as e:
                print(e)
                return ""

    def od(
        self,
        image,
    ):
        """
        Perform object detection on the given image.
        Args:
            image_path (str): The path to the image file on which to perform object detection.
        Returns:
            str: A JSON string containing the results of the object detection.
        """
        image = to_base64(image)
        json_str = self.invoke(
            prompt=self.od_desc,
            images=[image]
        )
        return json_str

    def od_v(
        self,
        image,
        save_path: str=None,
    ):
        """
        Perform object detection on an image and save the result.
        Args:
            image_path (str): The path to the input image.
            save_path (str): The path to save the output image with detected objects.
        Returns:
            tuple: A tuple containing the processed image and the save path.
        """
        json_str = self.od(image)
        img = draw_and_plot_boxes_from_json(json_str, image, save_path)
        return img, save_path


class MMChatter():
    def __init__(
        self,
        cli_dir: str,
        model_dir: str,
        mmproj_dir: str,
    ) -> None:
        """Initializes the class with the provided directories.

        Args:
            cli_dir (str): The directory for the CLI.
            model_dir (str): The directory for the model.
            mmproj_dir (str): The directory for the MMProj.

        Returns:
            None
        """
        self.cli_dir = cli_dir
        self.model_dir = model_dir
        self.mmproj_dir = mmproj_dir

    def get_resp(
        self,
        prompt: str,
        image: str,
        temp: float = 0.1,
        top_p: float = 0.8,
        top_k: int = 100,
        repeat_penalty: float = 1.05,
        n_context: int = 12800,
        n_max: int = 12800,
        ngl: int = 9999,
    ):
        """Get response from the model based on the given prompt and image.

        Args:
            prompt (str): The prompt to provide to the model.
            image (str): The image to be used as input for the model.
            temp (float, optional): Temperature parameter for sampling. Defaults to 0.1.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.8.
            top_k (int, optional): Top-k sampling parameter. Defaults to 100.
            repeat_penalty (float, optional): Repeat penalty for the model. Defaults to 1.05.

        Returns:
            str: The response generated by the model based on the input prompt and image.
        """
        # Define the command as a list of strings
        command = [
            self.cli_dir,
            "-m", self.model_dir,
            "--mmproj", self.mmproj_dir,
            "--image", image,
            "--temp", f"{temp}",
            "--top-p", f"{top_p}",
            "--top-k", f"{top_k}",
            "--repeat-penalty", f"{repeat_penalty}",
            "-c", f"{n_context}",
            "-n", f"{n_max}",
            "-p", prompt,
            "-ngl", f"{ngl}",
        ]

        # Use subprocess to run the command and capture the output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Process the output
        output = result.stdout

        # Extract the relevant part of the output
        # Locate the response starting from '<assistant>' and capture everything afterwards
        start_marker = "<assistant>"
        if start_marker in output:
            response_start = output.index(start_marker) + len(start_marker)
            response = output[response_start:].strip()
        else:
            response = output.strip()

        return response


def object_detect(
    image,
    prompt: str = OD_TEMPLATE,
    llm = None,
    cli_dir: str = None,
    model_dir: str = None,
    mmproj_dir: str = None,
    save_dir: str = None,
):
    resp = ""
    cli_dir = cli_dir if cli_dir is not None else os.getenv(
        "MM_CLI_DIR",
        None
    )
    if model_dir is None:
        model_dir = os.getenv(
            "MM_MODEL_DIR",
            "/home/jhu/dev/models/MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf"
        )
    if mmproj_dir is None:
        mmproj_dir = os.getenv(
            "MM_PROJ_DIR",
            "/home/jhu/dev/models/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"
        )
    if llm is None and cli_dir is None:
        raise ValueError("Either 'llm' or 'cli_dir' must be provided.")
    # mm_server = os.getenv("LLM_SERVER", "192.168.1.232")
    # mm_port = int(os.getevn("LLM_PORT", 22299))

    if cli_dir:
        mm = MMChatter(
            cli_dir=cli_dir,
            model_dir=model_dir,
            mmproj_dir=mmproj_dir
        )
        json_data = mm.get_resp(
            prompt=prompt,
            image=image
        )
    else:
        json_data = llm.od(image)
    _, save_dir = draw_and_plot_boxes_from_json(
        json_data=json_data,
        image=image,
        save_path=save_dir
    )
    resp += "Detected objects are:\n"
    resp += json_data
    resp += "\n"
    if save_dir:
        resp += "Picture with marks were saved at:\n"
        resp += save_dir
    return resp
