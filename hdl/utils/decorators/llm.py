import time
from functools import wraps

def measure_stream_performance(func):
    """
    Measures the performance of a streaming function by tracking the time taken to output characters and the rate of character and token generation.

    Args:
        func (callable): The streaming function to be measured. It should return a generator that yields dictionaries containing 'type' and 'content' keys.

    Returns:
        callable: A wrapper function that performs the performance measurement and prints statistics related to the streaming output.

    Statistics Printed:
        - Time to first character (in seconds)
        - Total time taken for the execution (in seconds)
        - Total number of characters output
        - Total number of tokens processed
        - Characters output per second
        - Tokens processed per second
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始计时
        start_time = time.time()

        # 执行流式处理函数并获取生成器
        stream = func(*args, **kwargs)

        # 初始化计时和统计变量
        first_char_time = None
        total_chars = 0
        total_tokens = 0

        for i in stream:
            total_tokens += 1
            if i['type'] == 'text' and i['content']:
                if first_char_time is None:
                    # 第一次输出字符的时间
                    first_char_time = time.time()

                # 输出字符并统计数量
                print(i['content'], end="", flush=True)
                total_chars += len(i['content'])

        # 总时间计算
        end_time = time.time()
        total_time = end_time - first_char_time
        time_to_first_char = first_char_time - start_time if first_char_time else total_time

        # 每秒输出的字符数
        chars_per_second = total_chars / total_time if total_time > 0 else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        # 打印统计信息
        print("\n--- Statistics ---")
        print(f"Time to first character: {time_to_first_char:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total characters: {total_chars}")
        print(f"Total tokens: {total_tokens}")
        print(f"Characters per second: {chars_per_second:.2f}")
        print(f"Tokens per second: {tokens_per_second:.2f}")

    return wrapper


@measure_stream_performance
def run_llm_stream(
    llm,
    client_id,
    prompt,
    **kwargs
):
    """
    Run a language model stream with the given parameters.

    Args:
        llm (object): The language model object used to generate responses.
        client_id (str): The unique identifier for the client making the request.
        prompt (str): The input prompt to which the language model should respond.
        **kwargs: Additional keyword arguments to customize the request.

    Returns:
        iterable: An iterable response stream from the language model.
    """
    resp = llm.stream(
        client_id=client_id,
        prompt=prompt,
        **kwargs
    )
    return resp