import yaml
import typing as t

from openai import OpenAI


class OpenAIWrapper(object):
    def __init__(
        self,
        client_conf: dict = None,
        client_conf_dir: str = None,
        load_conf: bool = True,
        *args,
        **kwargs
    ):
        """
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
        与上层保持完全兼容：内部改用 responses.create，
        但返回值/流式 chunk 仍然伪装成 chat.completions 的结构。
        """
        if not model:
            model = self.client_conf[client_id]['model']

        client = self.client_conf[client_id]['client']
        if response_model:
            import instructor
            client = instructor.from_openai(client)

        # ===== 构造 messages（保持你原有逻辑）=====
        messages = []
        if sys_info:
            messages.append({"role": "system", "content": sys_info})
        if history:
            messages.extend(history)

        # 规范化 media key（三元组）
        def _triple_keys(keys):
            if isinstance(keys, str):
                return (keys,)*3
            if len(keys) == 2:
                return (keys[0],) + tuple(keys)
            if len(keys) == 1:
                return (keys[0],)*3
            return keys

        image_keys = _triple_keys(image_keys)
        video_keys = _triple_keys(video_keys)

        content = [{"type": "text", "text": prompt}]
        if videos:
            if isinstance(videos, str):
                videos = [videos]
            for v in videos:
                content.append({
                    "type": video_keys[0],
                    video_keys[1]: {video_keys[2]: v}
                })

        if images:
            if isinstance(images, str):
                images = [images]
            for img in images:
                content.append({
                    "type": image_keys[0],
                    image_keys[1]: {image_keys[2]: img}
                })

        if (not images) and (not videos):
            content = prompt

        messages.append({"role": "user", "content": content})
        if assis_info:
            messages.append({"role": "assistant", "content": assis_info})

        # ===== Responses API 调用 =====
        # 注意：Responses 同时支持 messages 形状；tools 也直接传 tools / tool_choice。
        if stream:
            # --- 流式：返回一个生成器，伪装成 chat.completions 的 chunk 结构 ---
            #   你的上层 `for chunk in resp:` 会收到具有
            #   chunk.choices[0].delta.content / .tool_calls 的对象
            resp_stream = client.responses.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice=tool_choice if tools else None,
                stream=True,
                **kwargs
            )

            # 适配层：把 Responses 的事件流，转成 Chat Completions 风格的 chunk
            from types import SimpleNamespace
            def _wrap_delta_text(text):
                # -> chunk.choices[0].delta.content
                delta = SimpleNamespace(content=text)
                choice = SimpleNamespace(delta=delta)
                return SimpleNamespace(choices=[choice])

            def _wrap_delta_tool_call(name, arguments_fragment):
                # -> chunk.choices[0].delta.tool_calls[0].function.{name, arguments}
                func = SimpleNamespace(name=name, arguments=arguments_fragment)
                tool_call = SimpleNamespace(function=func)
                delta = SimpleNamespace(content=None, tool_calls=[tool_call])
                choice = SimpleNamespace(delta=delta)
                return SimpleNamespace(choices=[choice])

            def _generator():
                # SDK 的 Responses 流每个 event 有 event.type
                # 我们尽量覆盖主流事件名；未知事件直接忽略
                tool_args_acc = {}  # 累积每个工具参数（按 id 聚合）
                tool_name_cache = {}

                for event in resp_stream:
                    et = getattr(event, "type", None)

                    # 文本增量
                    if et == "response.output_text.delta":
                        delta_text = getattr(event, "delta", None)
                        if delta_text:
                            yield _wrap_delta_text(delta_text)

                    # 文本结束（可忽略，上层会基于yield的终止判断）
                    elif et == "response.output_text.done":
                        pass

                    # 工具调用参数增量
                    elif et in ("response.tool_call.delta", "response.function_call.delta"):
                        # 常见字段：event.id, event.name, event.delta / event.arguments_delta
                        call_id = getattr(event, "id", None)
                        name = getattr(event, "name", None) or tool_name_cache.get(call_id)
                        args_delta = getattr(event, "arguments_delta", None) or getattr(event, "delta", "")

                        if call_id:
                            tool_name_cache.setdefault(call_id, name or "")
                            tool_args_acc.setdefault(call_id, "")
                            tool_args_acc[call_id] += (args_delta or "")

                        # 也把这一小段增量向上抛（让你上层能尽快看到 tool_calls）
                        yield _wrap_delta_tool_call(name or "", args_delta or "")

                    # 工具调用完成（把完整参数再抛一次，便于上层一次性拿到）
                    elif et in ("response.tool_call.done", "response.function_call.done"):
                        call_id = getattr(event, "id", None)
                        full_name = tool_name_cache.get(call_id, "")
                        full_args = tool_args_acc.get(call_id, "")
                        yield _wrap_delta_tool_call(full_name, full_args)

                    # 其它事件（如 response.completed / response.error 等）
                    else:
                        # 可以按需扩展，这里静默忽略
                        pass

            return _generator()

        else:
            # --- 非流式：把 Responses 同步结果适配成 chat.completions 风格 ---
            resp = client.responses.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice=tool_choice if tools else None,
                stream=False,
                **kwargs
            )

            # 从 Responses 里抽取文本 & 工具调用
            # 尽量兼容：优先用 output_text；否则从 output 列表里聚合
            text_out = getattr(resp, "output_text", None)
            outputs = getattr(resp, "output", None)

            if text_out is None and outputs:
                # 聚合 message/output_text
                parts = []
                for item in outputs:
                    if getattr(item, "type", "") in ("message",):
                        # item.content 里通常还有若干块（output_text 等）
                        content_parts = getattr(item, "content", []) or []
                        for c in content_parts:
                            if getattr(c, "type", "") in ("output_text",):
                                parts.append(getattr(c, "text", ""))
                text_out = "".join(parts) if parts else None

            # 抽取工具调用（如果有）
            tool_calls_wrapped = []
            if outputs:
                for item in outputs:
                    if getattr(item, "type", "") in ("tool_call", "function_call"):
                        name = getattr(item, "name", "")
                        arguments = getattr(item, "arguments", "")
                        from types import SimpleNamespace
                        func = SimpleNamespace(name=name, arguments=arguments)
                        tool_calls_wrapped.append(SimpleNamespace(function=func))

            # 伪造 chat.completions 的返回结构
            from types import SimpleNamespace
            finish_reason = "tool_calls" if tool_calls_wrapped else "stop"
            message = SimpleNamespace(
                content=text_out or "",
                tool_calls=tool_calls_wrapped if tool_calls_wrapped else None
            )
            choice = SimpleNamespace(
                message=message,
                finish_reason=finish_reason
            )
            fake_resp = SimpleNamespace(choices=[choice])
            return fake_resp

    def get_resp_legacy(
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
            model = self.client_conf[client_id]['model']

        client = self.client_conf[client_id]['client']
        if response_model:
            import instructor #TODO 有些模型支持这个 instructor 的结构化输出，但实际上它调用的还是openai api的功能，以后适时删除或补全
            client = instructor.from_openai(client)

        messages = []

        if sys_info:
            messages.append({
                "role": "system",
                "content": sys_info
            })

        if history:
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