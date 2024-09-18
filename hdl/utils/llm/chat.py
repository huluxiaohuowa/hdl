import typing as t
import asyncio
from concurrent.futures import ProcessPoolExecutor


from openai import OpenAI
from ..desc.template import FN_TEMPLATE
from ..desc.func_desc import FN_DESC
import json
# import traceback


def chat_oai_stream(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy_key",
    model="default_model",
    prompt="Who are you?",
    *args,
    **kwargs
):
    """Chat with OpenAI's GPT-3 model using the specified parameters.

    Args:
        base_url (str): The base URL for the OpenAI API. Default is "http://127.0.0.1:8000/v1".
        api_key (str): The API key for accessing the OpenAI API. Default is "dummy_key".
        model (str): The model ID to use for the chat. Default is "/data/models/Qwen-7B-Chat-Int4".
        prompt (str): The initial prompt for the chat conversation.

    Yields:
        str: The generated content from the chat conversation.

    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        stream=True,
        *args,
        **kwargs
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        yield content


def chat_oai_invoke(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy_key",
    model="default_model",
    prompt="Who are you?",
    *args,
    **kwargs
):
    """Invoke OpenAI chat API to generate a response based on the given prompt.

    Args:
        base_url (str): The base URL of the OpenAI API. Default is "http://127.0.0.1:8000/v1".
        api_key (str): The API key for accessing the OpenAI API. Default is "dummy_key".
        model (str): The model to use for generating the response. Default is "/data/models/Qwen-7B-Chat-Int4".
        prompt (str): The prompt message to start the conversation. Default is "Who are you?".

    Returns:
        str: The response generated by the OpenAI chat API based on the prompt.
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        stream=False,
        *args,
        **kwargs
    )

    return response.choices[0].message.content

def run_tool_with_kwargs(tool, func_kwargs):
    """Run the specified tool with the provided keyword arguments.

    Args:
        tool (callable): The tool to be executed.
        func_kwargs (dict): The keyword arguments to be passed to the tool.

    Returns:
        The result of executing the tool with the provided keyword arguments.
    """
    return tool(**func_kwargs)


class OpenAI_M():
    def __init__(
        self,
        model_path: str = "default_model",
        device: str='gpu',
        generation_kwargs: dict = {},
        server_ip: str = "172.28.1.2",
        server_port: int = 8000,
        api_key: str = "dummy_key",
        tools: list = None,
        tool_desc: dict = None,
        *args,
        **kwargs
    ):
        """Initialize the OpenAI client with the specified parameters.

        Args:
            model_path (str): Path to the model (default is "default_model").
            device (str): Device to use (default is 'gpu').
            generation_kwargs (dict): Additional generation arguments (default is an empty dictionary).
            server_ip (str): IP address of the server (default is "172.28.1.2").
            server_port (int): Port of the server (default is 8000).
            api_key (str): API key for authentication (default is "dummy_key").
            tools (list): List of tools.
            tool_desc (dict): Description of tools.

        Raises:
            ValueError: If an invalid argument is provided.
        """
        # self.model_path = model_path
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{self.server_ip}:{str(self.server_port)}/v1"
        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            *args,
            **kwargs
        )
        self.tools = tools
        self.tool_desc = FN_DESC
        if tool_desc is not None:
            self.tool_desc = self.tool_desc | tool_desc

    def get_resp(
        self,
        prompt : str,
        images: list = [],
        image_keys: tuple = ("image", "image"),
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        model="default_model",
        stream: bool = True,
        **kwargs: t.Any,
    ):
        """Get response from chat completion model.

        Args:
            prompt (str): The prompt text to generate a response for.
            images (list, optional): List of image URLs to include in the prompt. Defaults to [].
            image_keys (tuple, optional): Tuple containing keys for image data. Defaults to ("image", "image").
            stop (list[str] | None, optional): List of strings to stop the conversation. Defaults to ["USER:", "ASSISTANT:"].
            model (str, optional): The model to use for generating the response. Defaults to "default_model".
            stream (bool, optional): Whether to stream the response or not. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the chat completion API.

        Yields:
            str: The generated response content.

        Returns:
            str: The generated response content if stream is False.
        """
        content = [
            {"type": "text", "text": prompt},
        ]
        if images:
            if isinstance(images, str):
                images = [images]
            for img in images:
                content.append({
                    "type": image_keys[0],
                    image_keys[0]: {
                        image_keys[1]: img
                    }
                })
        else:
            content = prompt

        response = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": content
            }],
            stream=stream,
            model=model,
            **kwargs
        )
        if stream:
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        else:
            return response.choices[0].message.content

    def invoke(
        self,
        *args,
        **kwargs
    ) -> str:
        """Invoke the function with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The response obtained by calling the get_resp method with the provided arguments and keyword arguments.
        """
        return self.get_resp(*args, stream=False, **kwargs)

    def stream(
        self,
        *args,
        **kwargs
    ):
        """Stream data from the server.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Response from the server with streaming enabled.
        """
        return self.get_resp(*args, stream=True, **kwargs)

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
        decision_dict = self.get_decision(prompt, **kwargs)
        if decision_dict.get("function_name", None) is None:
            return self.stream(prompt, **kwargs)
        else:
            tool_result = str(self.get_tool_result(prompt, **kwargs))
            prompt_final = "根据上下文回答最后的用户问题：\n上下文信息：\n"
            prompt_final += tool_result
            prompt_final += f"\n用户的问题：\n{prompt}"
            if stream:
                return self.stream(prompt_final, **kwargs)
            else:
                return self.invoke(prompt_final, **kwargs)

    def get_decision(
        self, prompt: str,
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
        for tool in self.tools:
            prompt_final += self.tool_desc.get(tool.__name__, "")
        prompt_final += f"\n用户的问题：\n{prompt}"
        decision_dict_str = self.invoke(prompt_final, **kwargs)
        print(decision_dict_str)
        return decision_dict_str

    def get_tool_result(
        self,
        prompt: str,
        **kwargs: t.Any
    ):
        """Get the result of a tool based on the decision made.

        Args:
            prompt (str): The prompt to make a decision.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the tool.
        """
        decision_dict_str = self.get_decision(
            prompt,
            **kwargs
        )
        try:
            decision_dict = json.loads(decision_dict_str)
        except Exception as e:
            print(e)
            return ""
        func_name = decision_dict.get("function_name", None)
        if func_name is None:
            return ""
        else:
            try:
                for tool in self.tools:
                    if tool.__name__ == func_name:
                        tool_final = tool
                func_kwargs = decision_dict.get("params")
                return tool_final(**func_kwargs)
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
            decision_dict = json.loads(decision_dict_str)
        except Exception as e:
            print(e)
            return ""
        func_name = decision_dict.get("function_name", None)
        if func_name is None:
            return ""
        else:
            try:
                for tool in self.tools:
                    if tool.__name__ == func_name:
                        tool_final = tool
                func_kwargs = decision_dict.get("params")

                loop = asyncio.get_running_loop()
                with ProcessPoolExecutor() as pool:
                    result = await loop.run_in_executor(pool, run_tool_with_kwargs, tool_final, func_kwargs)
                return result
            except Exception as e:
                print(e)
                return ""