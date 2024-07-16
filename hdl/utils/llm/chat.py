import typing as t

from llama_cpp import Llama
from jupyfuncs.llm.openapi import (
    chat_oai_invoke,
    chat_oai_stream
)


class GGUF_M(Llama):
    def __init__(
        self,
        model_path :str,
        device: str='gpu',
        generation_kwargs: dict = {},
        server_ip: str = "127.0.0.1",
        server_port: int = 8000,
        *args,
        **kwargs
    ):
        """Initialize the model with the specified parameters.
        
        Args:
            model_path (str): The path to the model.
            device (str, optional): The device to use, either 'gpu' or 'cpu'. Defaults to 'gpu'.
            generation_kwargs (dict, optional): Additional generation keyword arguments. Defaults to {}.
            server_ip (str, optional): The IP address of the server. Defaults to "127.0.0.1".
            server_port (int, optional): The port of the server. Defaults to 8000.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Raises:
            KeyError: If 'num_threads' or 'max_context_length' is missing in generation_kwargs.
        """
        print("正在从本地加载模型...")
        if device.lower() == 'cpu': 
            super().__init__(
                model_path=model_path,
                n_threads=generation_kwargs['num_threads'],
                n_ctx=generation_kwargs['max_context_length'],
                *args,
                **kwargs
            )
        else:
            super().__init__(
                model_path=model_path,
                n_threads=generation_kwargs['num_threads'],
                n_ctx=generation_kwargs['max_context_length'],
                n_gpu_layers=-1,
                flash_attn=True,
                *args,
                **kwargs
            )
        self.generation_kwargs = generation_kwargs

    def invoke(
        self, 
        prompt : str,
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        # history: list = [],
        **kwargs: t.Any,
    ) -> str:
        """Invoke the model to generate a response based on the given prompt.
        
        Args:
            prompt (str): The prompt to be used for generating the response.
            stop (list[str], optional): List of strings that indicate when the model should stop generating the response. Defaults to ["USER:", "ASSISTANT:"].
            **kwargs: Additional keyword arguments to be passed to the model.
        
        Returns:
            str: The generated response based on the prompt.
        """
        prompt_final = f"USER:\n{prompt}\nASSISTANT:\n" 

        result = self.create_completion(
            prompt_final,
            repeat_penalty=self.generation_kwargs["repetition_penalty"],
            max_tokens=self.generation_kwargs["max_new_tokens"],
            stop=stop,
            echo=False,
            temperature=self.generation_kwargs["temperature"],
            mirostat_mode = 2,
            mirostat_tau=4.0,
            mirostat_eta=1.1
        )
        resp = result['choices'][0]['text']
        # history.append(
        #     [prompt, resp]
        # )
        return resp

    def stream(
        self,
        prompt: str,
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        # history: list = [],
        **kwargs: t.Any,
    ):
        """Generate text responses based on the given prompt using the model.
        
        Args:
            prompt (str): The prompt to generate text responses.
            stop (list[str], optional): List of strings to stop the generation. Defaults to ["USER:", "ASSISTANT:"].
            **kwargs: Additional keyword arguments for the model.
        
        Yields:
            str: Text responses generated by the model based on the prompt.
        """
        prompt = f"USER:\n{prompt}\nASSISTANT:\n" 
        output = self.create_completion(
            prompt,
            stream=True,
            repeat_penalty=self.generation_kwargs["repetition_penalty"], 
            max_tokens=self.generation_kwargs["max_new_tokens"],
            stop=stop,
            echo=False,
            temperature=self.generation_kwargs["temperature"],
            mirostat_mode = 2,
            mirostat_tau=4.0,
            mirostat_eta=1.1
        )
        # history.append([])
        for chunk in output:
            item = chunk['choices'][0]['text']
            # self.resps[-1].append(item)
            yield chunk['choices'][0]['text']
        # self.resps[-1] = "".join(self.resps[-1])


# class GGUF_M():
#     def __init__(
#         self,
#         model_path :str,
#         device: str='gpu',
#         generation_kwargs: dict = {},
#         server_ip: str = "127.0.0.1",
#         server_port: int = 8000,
#     ):
#         """Initialize the model with the provided model path and optional parameters.
        
#         Args:
#             model_path (str): The path to the model.
#             device (str, optional): The device to use for model initialization. Defaults to 'gpu'.
#             generation_kwargs (dict, optional): Additional keyword arguments for model generation. Defaults to {}.
#             server_ip (str, optional): The IP address of the server. Defaults to "127.0.0.1".
#             server_port (int, optional): The port of the server. Defaults to 8000.
#         """ 
#         # 从本地初始化模型
#         # super().__init__()
#         self.generation_kwargs = generation_kwargs
#         print("正在从本地加载模型...")
#         if device == 'cpu':
#             self.model = Llama(
#                 model_path=model_path,
#                 n_threads=self.generation_kwargs['num_threads'],
#                 n_ctx=self.generation_kwargs['max_context_length'],
#             )
#         else:
#             self.model = Llama(
#                 model_path=model_path,
#                 n_threads=self.generation_kwargs['num_threads'],
#                 n_ctx=self.generation_kwargs['max_context_length'],
#                 n_gpu_layers=-1,
#                 flash_attn=True
#             )
            
#         print("完成本地模型的加载")

#     def invoke(
#         self, 
#         prompt : str,
#         stop: list[str] | None = ["USER:", "ASSISTANT:"],
#         # history: list = [],
#         **kwargs: t.Any,
#     ) -> str:
#         """Invoke the model to generate a response based on the given prompt.
        
#         Args:
#             prompt (str): The prompt to be used for generating the response.
#             stop (list[str], optional): List of strings that indicate when the model should stop generating the response. Defaults to ["USER:", "ASSISTANT:"].
#             **kwargs: Additional keyword arguments to be passed to the model.
        
#         Returns:
#             str: The generated response based on the prompt.
#         """
#         prompt_final = f"USER:\n{prompt}\nASSISTANT:\n" 

#         result = self.model.create_completion(
#             prompt_final,
#             repeat_penalty=self.generation_kwargs["repetition_penalty"],
#             max_tokens=self.generation_kwargs["max_new_tokens"],
#             stop=stop,
#             echo=False,
#             temperature=self.generation_kwargs["temperature"],
#             mirostat_mode = 2,
#             mirostat_tau=4.0,
#             mirostat_eta=1.1
#         )
#         resp = result['choices'][0]['text']
#         # history.append(
#         #     [prompt, resp]
#         # )
#         return resp
 
#     def stream(
#         self,
#         prompt: str,
#         stop: list[str] | None = ["USER:", "ASSISTANT:"],
#         # history: list = [],
#         **kwargs: t.Any,
#     ):
#         """Generate text responses based on the given prompt using the model.
        
#         Args:
#             prompt (str): The prompt to generate text responses.
#             stop (list[str], optional): List of strings to stop the generation. Defaults to ["USER:", "ASSISTANT:"].
#             **kwargs: Additional keyword arguments for the model.
        
#         Yields:
#             str: Text responses generated by the model based on the prompt.
#         """
#         prompt = f"USER:\n{prompt}\nASSISTANT:\n" 
#         output = self.model.create_completion(
#             prompt,
#             stream=True,
#             repeat_penalty=self.generation_kwargs["repetition_penalty"], 
#             max_tokens=self.generation_kwargs["max_new_tokens"],
#             stop=stop,
#             echo=False,
#             temperature=self.generation_kwargs["temperature"],
#             mirostat_mode = 2,
#             mirostat_tau=4.0,
#             mirostat_eta=1.1
#         )
#         # history.append([])
#         for chunk in output:
#             item = chunk['choices'][0]['text']
#             # self.resps[-1].append(item)
#             yield chunk['choices'][0]['text']
#         # self.resps[-1] = "".join(self.resps[-1])


class OpenAI_M():
    def __init__(
        self,
        model_path: str = None,
        device: str='gpu',
        generation_kwargs: dict = {},
        server_ip: str = "172.28.1.2",
        server_port: int = 8000,
    ):
        """Initialize the class with the specified parameters.
        
        Args:
            model_path (str, optional): Path to the model file. Defaults to None.
            device (str, optional): Device to run the model on. Defaults to 'gpu'.
            generation_kwargs (dict, optional): Additional keyword arguments for model generation. Defaults to {}.
            server_ip (str, optional): IP address of the server. Defaults to "172.28.1.2".
            server_port (int, optional): Port number of the server. Defaults to 8000.
        """
        self.model_path = model_path
        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = "http://{self.server_ip}:{str(self.server_port)}/v1"
    
    def invoke(
        self, 
        prompt : str,
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        # history: list = [],
        **kwargs: t.Any,
    ) -> str:
        """Invoke the chatbot with the given prompt and return the response.
        
        Args:
            prompt (str): The prompt to provide to the chatbot.
            stop (list[str], optional): List of strings that indicate the end of the conversation. Defaults to ["USER:", "ASSISTANT:"].
            **kwargs: Additional keyword arguments to pass to the chatbot.
        
        Returns:
            str: The response generated by the chatbot.
        """
        resp = chat_oai_invoke(
            base_url=self.base_url,
            model=self.model_path,
            prompt=prompt
        )
        return resp
    
    def stream(
        self, 
        prompt : str,
        stop: list[str] | None = ["USER:", "ASSISTANT:"],
        # history: list = [],
        **kwargs: t.Any,
    ) -> str:
        """Generate a response by streaming conversation with the OpenAI chat model.
        
        Args:
            prompt (str): The prompt to start the conversation.
            stop (list[str], optional): List of strings that indicate when the conversation should stop. Defaults to ["USER:", "ASSISTANT:"].
            **kwargs: Additional keyword arguments to pass to the chat model.
        
        Returns:
            str: The response generated by the chat model.
        """
        resp = chat_oai_stream(
            base_url=self.base_url,
            model=self.model_path,
            prompt=prompt
        )
        return resp