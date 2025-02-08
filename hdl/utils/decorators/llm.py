import time
from functools import wraps

def measure_stream_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始计时
        start_time = time.time()

        # 执行流式处理函数并获取生成器
        stream = func(*args, **kwargs)

        # 初始化计时和统计变量
        first_char_time = None
        total_chars = 0

        for i in stream:
            if i['type'] == 'text' and i['content']:
                if first_char_time is None:
                    # 第一次输出字符的时间
                    first_char_time = time.time()

                # 输出字符并统计数量
                print(i['content'], end="", flush=True)
                total_chars += len(i['content'])

        # 总时间计算
        end_time = time.time()
        total_time = end_time - start_time
        time_to_first_char = first_char_time - start_time if first_char_time else total_time

        # 每秒输出的字符数
        chars_per_second = total_chars / total_time if total_time > 0 else 0

        # 打印统计信息
        print("\n--- Statistics ---")
        print(f"Time to first character: {time_to_first_char:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Characters per second: {chars_per_second:.2f}")

    return wrapper


@measure_stream_performance
def run_llm_stream(
    llm,
    client_id,
    prompt,
    **kwargs
):
    resp = llm.stream(
        client_id=client_id,
        prompt=prompt,
        **kwargs
    )
    return resp