# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/llm_wrapper.py
# 说明：大模型调用封装
# 模块功能：OpenAIWrapper 按 YAML/字典配置同时初始化多个 OpenAI 协议客户端（对话与文本嵌入均可），
#           统一支持多轮历史、图片与视频注入消息体、函数调用（function calling）与结构化输出解析。
import yaml
import typing as t

from openai import OpenAI


class OpenAIWrapper(object):
    """按配置文件同时初始化多个 OpenAI 协议客户端的包装类：client_conf 是 client_id -> {host, model, api_key, client_type, client} 的字典，
    每次调用用 client_id 选路，避免为每个模型/服务单独建对象。
    关键属性：client_conf（配置与客户端实例）、client_conf_path（YAML 路径，load_clients 用）。
    典型用法：add_client/load_clients 配置好后，用 get_resp 取原始响应，invoke/stream 取解析后的文本或函数调用（function calling）参数，embedding 取文本嵌入（text embedding）向量。"""
    def __init__(
        self,
        client_conf: dict = None,
        client_conf_dir: str = None,
        load_conf: bool = True,
        *args,
        **kwargs
    ):
        """
        配置来源二选一：client_conf 字典直接用，或 client_conf_dir 指向 YAML 且 load_conf=True 时读文件；
        随后为每条配置创建一个 OpenAI 客户端挂到 conf["client"]，缺 api_key 时用占位串，并把缺省 client_type 记为 "chat"。

        Initializes the client configuration for the class.

        Args:
            client_conf (dict, optional): A dictionary containing client configuration. If None,
                client configuration will be loaded from the specified directory.
            client_conf_dir (str, optional): The directory from which to load client configuration
                if `client_conf` is None. Must be provided in that case.
            load_conf (bool, optional): A flag indicating whether to load the client
                configuration from the directory. Defaults to True.
            *args: Variable length argument list for client initialization.
            **kwargs: Arbitrary keyword arguments for client initialization.

        Raises:
            AssertionError: If `client_conf` is None and `client_conf_dir` is also None.

        Note:
            The method will create a client for each configuration found in `client_conf`,
            initializing the client with the specified `base_url` and `api_key`.
        Examples:
            >>> llm = OpenAIWrapper(
            >>>     client_conf_dir="/some/path/model_conf.yaml",
            >>>     # load_conf=False
            >>> )
)
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
            if "client_type" not in conf:
                conf["client_type"] = "chat"

    def add_client(
        self,
        client_id: str,
        host: str,
        port: int = None,
        model: str = "default_model",
        api_key: str = "dummy_key",
        **kwargs
    ):
        """
        运行时追加一个客户端：host 不以 http 开头且给了 port 时拼成 http://host:port/v1，并把 kwargs 里的 client_type 弹出记为 "chat"（其余 kwargs 交给 OpenAI 客户端）。

        Add a new client configuration to the client manager.

        This method stores the configuration details for a new client identified by the
        provided client ID. It constructs the host URL based on the input parameters
        and initializes an OpenAI client instance.

        Args:
            client_id (str): Unique identifier for the client.
            host (str): Hostname or IP address of the client.
            port (int, optional): Port number for the client connection. Defaults to None.
            model (str, optional): Model to use for the client. Defaults to "default_model".
            api_key (str, optional): API key for authentication. Defaults to "dummy_key".
            **kwargs: Additional keyword arguments passed to the OpenAI client.

        Raises:
            ValueError: If both host and port are not valid for constructing a URL.
        Examples:
            >>> llm.add_client(
            >>>     client_id="rone",
            >>>     host="127.0.0.1",
            >>>     port=22299,
            >>>     model="ictrek/rone:1.5b32k",
            >>> )
        """
        self.client_conf[client_id] = {}
        if not host.startswith('http') and port:
            host = f"http://{host}:{port}/v1"
        self.client_conf[client_id]['host'] = host
        self.client_conf[client_id]['model'] = model
        self.client_conf[client_id]['client_type'] = kwargs.pop("client_type", "chat")
        self.client_conf[client_id]['client'] = OpenAI(
            base_url=host,
            api_key=api_key,
            **kwargs
        )

    def load_clients(self):
        """
        读 self.client_conf_path 指向的 YAML，把每条配置的 host/port 规范化成 http://host:port/v1（OpenAI 兼容服务的版本路径）后写回 self.client_conf。
        注意：本方法只改配置字典，OpenAI 客户端实例由 __init__ 末尾的循环创建，因此运行时单独调用它不会补建 client 键。

        Loads client configuration from a YAML file and updates the 'host' field
        for each client entry, ensuring the correct URL format.

        This method reads the client configuration from the specified path,
        updates the 'host' field to include the appropriate port and the
        'http' protocol if not already specified, and stores the updated
        configuration in the `client_conf` attribute.

        Attributes:
            client_conf_path (str): The file path to the client configuration YAML file.
            client_conf (dict): The updated client configuration after processing.

        Returns:
            None
        """
        with open(self.client_conf_path, 'r') as file:
            data = yaml.safe_load(file)

        # 更新 host 字段
        for _, value in data.items():
            host = value.get('host', '')
            port = value.get('port', '')
            if not host.startswith('http') and port:  # 确保有 port 才处理
                value['host'] = f"http://{host}:{port}/v1"
        self.client_conf = data

    def get_resp(
        self,
        prompt,
        client_id: str = None,
        history: list = None,
        sys_info: str = None,
        assis_info: str = None,
        images: list = None,
        image_keys: tuple = ("image_url", "url"),
        videos: list = None,
        video_keys: tuple = ("video_url", "url"),
        model: str=None,
        tools: list = None,
        tool_choice: str = "auto",
        stream: bool = True,
        response_model = None,
        **kwargs: t.Any,
    ):
        """
        按 client_id 取客户端拼消息并请求：顺序为 system → history 原样展开 → user（多模态分块或纯文本）→ 末条 assistant 预答（assis_info），
        有 tools 时带上 tool_choice（默认 auto）让模型自行决定是否函数调用（function calling）；response_model 非空时先用 instructor 包装客户端以解析结构化输出。
        Returns: 底层 chat.completions.create 的原生响应对象；stream=True 为分块迭代器，False 为完整响应。

        Generates a response from a chat model based on the given prompt and additional context.

        Args:
            prompt (str): The main text prompt to send to the chat model.
            client_id (str, optional): Identifier for the client configuration. Defaults to None.
            history (list, optional): A list of previous messages to provide context for the conversation. Each message should be a dictionary with "role" and "content". Defaults to None.
            sys_info (str, optional): System-level information to set the context of the chat. Defaults to None.
            assis_info (str, optional): Information from the assistant to be included in the conversation. Defaults to None.
            images (list, optional): A list of images to include in the message content. Defaults to None.
            image_keys (tuple, optional): Keys to format the image data. Must be of length 1 or 2. Defaults to ("image_url", "url").
            model (str, optional): The model to use for generating the response. If not provided, it defaults to the one in client configuration for the given client_id.
            tools (list, optional): List of tools to be available during the chat. Defaults to None.
            stream (bool, optional): Whether to stream the response. Defaults to True.
            response_model (optional): Specifies the response model to use. Defaults to None.
            **kwargs (Any): Additional configuration parameters.

        Returns:
            Response: The response object from the chat model.
        """
        if not model:
            # 模型名缺省取该 client_id 配置里的 model
            model = self.client_conf[client_id]['model']

        client = self.client_conf[client_id]['client']
        if response_model:
            # 结构化输出（structured output）：只把客户端包成 instructor 客户端，response_model 本身未传给 create，故仍按原样返回响应
            import instructor #TODO 有些模型支持这个 instructor 的结构化输出，但实际上它调用的还是openai api的功能，以后适时删除或补全
            client = instructor.from_openai(client)

        messages = []

        if sys_info:
            messages.append({
                "role": "system",
                "content": sys_info
            })

        if history:
            # 多轮上下文：调用方给的 role/content 字典列表原样接在 system 之后，本函数不做裁剪或 token 计数
            messages.extend(history)
            # history 需要符合以下格式，其中system不是必须
            # history = [
            #     {"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user", "content": "message 1 content."},
            #     {"role": "assistant", "content": "message 2 content"},
            #     {"role": "user", "content": "message 3 content"},
            #     {"role": "assistant", "content": "message 4 content."},
            #     {"role": "user", "content": "message 5 content."}
            # ]

        if not model:
            model = self.client_conf[client_id]["model"]
        # Adjust the image_keys to be a tuple of length 3 based on its current length
        if isinstance(image_keys, str):
            image_keys = (image_keys,) * 3
        elif len(image_keys) == 2:
            image_keys = (image_keys[0],) + tuple(image_keys)
        elif len(image_keys) == 1:
            image_keys = (image_keys[0],) * 3

        if isinstance(video_keys, str):
            video_keys = (video_keys,) * 3
        elif len(video_keys) == 2:
            video_keys = (video_keys[0],) + tuple(video_keys)
        elif len(video_keys) == 1:
            video_keys = (video_keys[0],) * 3

        content = [{
            "type": "text",
            "text": prompt
        }]

        if videos:
            if isinstance(videos, str):
                images = [videos]
            for video in videos:
                content.append({
                    "type": video_keys[0],
                    video_keys[1]: {
                        video_keys[2]: video
                    }
                })


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
        if (not images) and (not videos):
            content = prompt

        # Add the user's input as a message
        messages.append({
            "role": "user",
            "content": content
        })

        if assis_info:
            messages.append({
                "role": "assistant",
                "content": assis_info
            })

        if tools:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                **kwargs
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )
        return resp

    def invoke(
        self,
        prompt,
        **kwargs
    ):
        """
        Invoke the API to get a response based on the provided prompt.

        Args:
            prompt (str): The input prompt to be processed.
            **kwargs: Additional keyword arguments to customize the API request.

        Returns:
            dict: A dictionary containing the type of response and its contents.
                  The possible keys are:
                  - 'type' (str): Indicates the type of response ('text' or 'tool_calls').
                  - 'contents' (str, optional): The text content if the response type is 'text'.
                  - 'tool_params' (dict, optional): The parameters of the tool called if the response type is 'tool_calls'.

        Examples:
            >>> llm.invoke(
            >>>     client_id="glm_4_flash",
            >>>     prompt="深圳天气怎么样？",
            >>>     tools=[TOOL_DICT['get_weather']],
            >>> )
            {'type': 'tool_calls',
             'tool_parmas': Function(arguments='{"location": "Shenzhen"}', name='get_weather')}
        """
        answer_dict = {}

        resp = self.get_resp(
            prompt,
            stream=False,
            **kwargs
        )
        if resp.choices[0].finish_reason == "stop":
            answer_dict["type"] = "text"
            answer_dict["contents"] = resp.choices[0].message.content
        elif resp.choices[0].finish_reason == "tool_calls":
            answer_dict["type"] = "tool_calls"
            answer_dict["tool_params"] = resp.choices[0].message.tool_calls[0].function

        return answer_dict

    def stream(self, prompt, **kwargs):
        """
        Streams responses based on the provided prompt, yielding chunks of data.

        This function calls the `get_resp` method with the prompt and additional keyword arguments,
        streaming the response in chunks. It processes each chunk to yield either tool call parameters
        or textual content. If an error occurs while processing the chunks, it yields an error message.

        Args:
            prompt (str): The input prompt to generate responses for.
            **kwargs: Additional keyword arguments to be passed to the `get_resp` method.

        Yields:
            dict: A dictionary with the following possible keys:
                - type (str): Indicates the type of the response ('tool_calls', 'text', or 'error').
                - tool_params (dict, optional): Parameters of the tool call if the type is 'tool_calls'.
                - content (str, optional): The generated text content if the type is 'text'.
                - message (str, optional): An error message if the type is 'error'.

        Examplse:
            >>> resp = llm.stream(
            >>>     client_id="r1", #此模型可以进行cot
            >>>     prompt=prompt,
            >>>     # tools=[TOOL_DICT['get_weather']],
            >>> )
            >>> for i in resp:
            >>>     if i['type'] == 'text' and i['content']:
            >>>         print(i['content'], flush=True, end="")
        """
        resp = self.get_resp(prompt=prompt, stream=True, **kwargs)

        for chunk in resp:
            try:
                choice = chunk.choices[0]

                # 如果返回了 tool_calls
                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                    tool_calls = choice.delta.tool_calls
                    if tool_calls:  # 防止为空
                        yield {
                            "type": "tool_calls",
                            "tool_params": tool_calls[0].function
                        }
                    return  # 直接返回，结束流式输出

                # 返回文本内容
                elif hasattr(choice.delta, 'content'):
                    yield {
                        "type": "text",
                        "content": choice.delta.content
                    }

            except (AttributeError, IndexError) as e:
                # 防止意外的结构异常
                yield {
                    "type": "error",
                    "message": f"Stream chunk error: {str(e)}"
                }
                return

        return

    def embedding(
        self,
        client_id: str,
        texts: list[str],
        model: str = None,
        **kwargs
    ):
        """
        Generates embeddings for a list of texts using a specified model.

        Args:
            client_id (str): The ID of the client to use for generating embeddings.
            texts (list[str]): A list of texts for which to generate embeddings.
            model (str, optional): The model to use for generating embeddings.
                If not provided, the model specified in the client configuration will be used.
            **kwargs: Additional keyword arguments to be passed to the client embedding creation method.

        Returns:
            list: A list of embeddings corresponding to the input texts.
        """
        if not model:
            model = self.client_conf[client_id]['model']

        client = self.client_conf[client_id]['client']
        response = client.embeddings.create(
            input=texts,
            model=model,
            **kwargs
        )

        return [i.embedding for i in response.data]