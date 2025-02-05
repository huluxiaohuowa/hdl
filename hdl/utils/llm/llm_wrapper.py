import yaml
import typing as t

from openai import OpenAI
import instructor


class OpenAIWrapper(object):
    def __init__(
        self,
        client_conf: dict = None,
        client_conf_dir: str = None,
        load_conf: bool = True,
        *args,
        **kwargs
    ):
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

    def get_resp(
        self,
        prompt,
        client_id: str = None,
        history: list = None,
        sys_info: str = None,
        assis_info: str = None,
        images: list = None,
        image_keys: tuple = ("image_url", "url"),
        model: str=None,
        tools: list = None,
        stream: bool = True,
        response_model = None,
        **kwargs: t.Any,
    ):
        if not model:
            model = self.client_conf[client_id]['model']

        client = self.client_conf[client_id]['client']
        if response_model:
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

        content = []
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

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream
        )
        return resp


