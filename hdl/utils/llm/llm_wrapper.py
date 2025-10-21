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
        """
        self.client_conf = {}
        if client_conf is None:
            assert client_conf_dir is not None
            self.client_conf_path = client_conf_dir
            if load_conf:
                self.load_clients()
        else:
            self.client_conf = client_conf

        for cid, conf in self.client_conf.items():
            conf["client"] = OpenAI(
                base_url=conf["host"],
                api_key=conf.get("api_key", "dummy_key"),
                *args,
                **kwargs
            )
            if "client_type" not in conf:
                conf["client_type"] = "chat"
            if "model" not in conf:
                conf["model"] = None

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
        self.client_conf[client_id]['client_type'] = kwargs.pop("client_type", "chat")
        self.client_conf[client_id]['client'] = OpenAI(
            base_url=host,
            api_key=api_key,
            **kwargs
        )

    def load_clients(self):
        with open(self.client_conf_path, 'r') as file:
            data = yaml.safe_load(file)
        for _, value in data.items():
            host = value.get('host', '')
            port = value.get('port', '')
            if not host.startswith('http') and port:
                value['host'] = f"http://{host}:{port}/v1"
        self.client_conf = data

    def get_resp(
        self,
        prompt: str,
        client_id: str = None,
        history: list = None,
        sys_info: str = None,
        assis_info: str = None,
        images: list = None,
        image_keys: tuple = ("image_url", "url"),
        videos: list = None,
        video_keys: tuple = ("video_url", "url"),
        model: str = None,
        tools: list = None,
        tool_choice: str = "auto",
        stream: bool = True,
        response_model = None,
        **kwargs: t.Any,
    ):
        """
        Generates a response from the model using responses.create with input=.
        Supports optional image input if `images` is provided.
        Also supports external tools via `tools` + `tool_choice`.
        """
        if client_id is None:
            raise ValueError("client_id must be provided")
        conf = self.client_conf[client_id]
        client = conf["client"]
        if model is None:
            model = conf.get("model")
        if model is None:
            raise ValueError("model must be specified either in client_conf or via parameter")

        # Build input list
        input_items: list[t.Any] = []

        if sys_info:
            input_items.append({"role": "system", "content": sys_info})
        if history:
            input_items.extend(history)

        # Build user message
        if images:
            if isinstance(images, str):
                images = [images]
            multimodal_content = [
                {"type": "input_text", "text": prompt}
            ]
            for img in images:
                multimodal_content.append({
                    "type": "input_image",
                    "image_url": img
                })
            user_item = {"role": "user", "content": multimodal_content}
        else:
            user_item = {"role": "user", "content": prompt}

        input_items.append(user_item)

        if assis_info:
            input_items.append({"role": "assistant", "content": assis_info})

        # Prepare call parameters
        call_params = {
            "model": model,
            "input": input_items,
            **kwargs
        }
        if tools:
            call_params["tools"] = tools
            call_params["tool_choice"] = tool_choice

        # Call Responses API
        if stream:
            resp_stream = client.responses.create(
                stream=True,
                **call_params
            )
            return resp_stream
        else:
            resp = client.responses.create(
                stream=False,
                **call_params
            )
            # Wrap to mimic chat.completions interface
            from types import SimpleNamespace
            text_out = getattr(resp, "output_text", "")
            message = SimpleNamespace(content=text_out, tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            fake_resp = SimpleNamespace(choices=[choice])
            return fake_resp

    def invoke(
        self,
        prompt: str,
        **kwargs
    ):
        answer_dict = {}
        resp = self.get_resp(prompt=prompt, stream=False, **kwargs)
        answer_dict["type"] = "text"
        answer_dict["contents"] = resp.choices[0].message.content
        return answer_dict

    def stream(self, prompt: str, **kwargs):
        resp_stream = self.get_resp(prompt=prompt, stream=True, **kwargs)
        for event in resp_stream:
            delta = getattr(event, "delta", None)
            if delta and hasattr(delta, "content"):
                yield {"type": "text", "content": delta.content}
        return

    def embedding(
        self,
        client_id: str,
        texts: list[str],
        model: str = None,
        **kwargs
    ) -> list:
        if model is None:
            model = self.client_conf[client_id]['model']
        client = self.client_conf[client_id]['client']
        response = client.embeddings.create(
            input=texts,
            model=model,
            **kwargs
        )
        return [i.embedding for i in response.data]