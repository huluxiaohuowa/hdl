# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/weather/weather.py
# 说明：天气查询工具
import requests
from pathlib import Path
import os
import json

from bs4 import BeautifulSoup
import numpy as np

from ..llm.embs import HFEmbedder


def get_city_codes():
    """Get city codes from a JSON file.
    读取随包的 datasets/city_code.json（相对本文件上溯三级目录定位，与运行时工作目录无关），返回 {城市名: 天气网城市编号} 字典（键为中文城市名，值为整数编号，只读不写）。

    Returns:
        dict: A dictionary containing city codes.
    """
    # with open('../../city.json', 'r', encoding='utf-8') as f:
    #     code_dic = eval(f.read())
    # return code_dic
    code_file = Path(__file__).resolve().parent.parent.parent \
        / "datasets" \
        / "city_code.json"
    with code_file.open() as f:
        codes = json.load(f)
    return codes


def get_html(code):
    """Get the HTML content of a weather webpage based on the provided code.
    按城市编号抓取中国天气网的七天天气页：请求 http://www.weather.com.cn/weather/<code>.shtml（code 即 get_city_codes() 字典的值），带桌面 Chrome 的 User-Agent 头以躲过基本的 UA 过滤，并把响应强制按 utf-8 解码后返回页面 HTML 文本；请求本身会打印 URL，未设超时，网络失败直接抛 requests 异常。

    Args:
        code (str): The code used to identify the specific weather webpage.

    Returns:
        str: The HTML content of the weather webpage.

    Example:
        html_content = get_html('101010100')
    """
    weather_url = f'http://www.weather.com.cn/weather/{code}.shtml'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"}
    print(weather_url)
    resp = requests.get(url=weather_url, headers=header)
    resp.encoding = 'utf-8'
    return resp.text


def get_page_data(html):
    """Get weather information for the next seven days from the provided HTML content.

    Args:
        html (str): The HTML content containing weather information.

    Returns:
        str: A formatted string with weather information for the next seven days.
    """
    soup = BeautifulSoup(html, 'html.parser')
    weather_info = soup.find('div', id='7d')
    seven_weather = weather_info.find('ul')
    weather_list = seven_weather.find_all('li')

    weather_str = ""

    for weather in weather_list:
        # print("\n")
        weather_str += (weather.find('h1').get_text() + "\n") # 日期
        weather_str += ('天气状况：' + weather.find('p', class_='wea').get_text() + "\n")
        # 判断标签'p','tem'下是否有标签'span'，以此判断是否有最高温
        if weather.find('p', class_='tem').find('span'):
            temp_high = weather.find('p', class_='tem').find('span').get_text()
        else:
            temp_high = ''  # 最高温
        temp_low = weather.find('p', class_='tem').find('i').get_text()  # 最低温
        weather_str += (f'天气温度：{temp_low}/{temp_high}' + "\n")
        win_list_tag = weather.find('p', class_='win').find('em').find_all('span')
        win_list = []
        for win in win_list_tag:
            win_list.append(win.get('title'))
        weather_str += ('风向：' + '-'.join(win_list) + "\n")
        weather_str += ('风力：' + weather.find('p', class_='win').find('i').get_text() + "\n")
        weather_str += "\n"

    return weather_str


def get_weather(city):
    """Get the weather information for a specific city.
    按城市名查天气并返回拼好的中文文本：先查 get_city_codes()，命中就用该城市的天气网编号抓页面；未命中时调用 get_standard_cityname（加载嵌入模型做向量近邻）取最相近的标准城市名，并在结果开头附一行「识别为…」的提示；
    随后 get_html 抓页、get_page_data 解析成逐日预报，末尾附上「标准城市名(原名)」的标题行，返回完整字符串；
    近邻匹配总能返回某个键（argmax 不设阈值），故拼写有误的城市会被强行认作最相近的城市；模型目录缺失或网络不通时在 HFEmbedder 或抓页处直接抛异常，本函数不做兜底。

    Args:
        city (str): The name of the city to get weather information for.

    Returns:
        str: A string containing the latest weather information for the specified city.
    """
    code_dic = get_city_codes()
    city_name = city
    weather_str = ""
    if city not in code_dic:
        city_name = get_standard_cityname(city)
        weather_str += f"{city}识别为{city_name}，若识别错误，请提供更为准确的城市名\n"
    html = get_html(code_dic[city_name])
    result = get_page_data(html)
    weather_str += f"\n{city_name}({city})的最新查到的天气信息如下：\n\n"
    weather_str += result
    return weather_str


def get_standard_cityname(
    city,
    emb_dir: str = os.getenv(
        'EMB_MODEL_DIR',
        '/home/jhu/dev/models/bge-m3'
    )
):
    """Get the standard city name based on the input city name.
    用文本嵌入做城市名近邻归一：从随包的 datasets/city_embs.npy 读入预存的城市名向量矩阵，用 HFEmbedder（SentenceTransformer，权重目录取 emb_dir，由环境变量 EMB_MODEL_DIR 决定默认值，每次调用都重新加载模型并转半精度）编码输入的 city，
    城市向量与查询向量做内积得相似度，返回 code_dic 键序上 argmax 对应的标准城市名；
    前提是该 npy 的行序与 city_code.json 的键序严格一致，否则返回的名字与编号错位。

    Args:
        city (str): The input city name.
        emb_dir (str): The directory path for the embedding model (default is '/home/jhu/dev/models/bge-m3').

    Returns:
        str: The standard city name.
    """
    code_dic = get_city_codes()
    city_list = list(code_dic.keys())

    city_embs = np.load(
        Path(__file__).resolve().parent.parent.parent \
            / "datasets" \
            / "city_embs.npy"
    )

    emb = HFEmbedder(
        emb_dir=emb_dir,
    )
    query_emb = emb.encode(city)
    sims = city_embs @ query_emb.T

    return city_list[np.argmax(sims)]
