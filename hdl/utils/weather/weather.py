import requests
from pathlib import Path
import os
import json

from bs4 import BeautifulSoup
import numpy as np

from ..llm.embs import HFEmbedder


def get_city_codes():
    """Get city codes from a JSON file.

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
