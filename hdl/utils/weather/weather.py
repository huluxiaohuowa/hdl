import requests
from pathlib import Path
import os
import json
from bs4 import BeautifulSoup


def get_city_codes():
    # with open('../../city.json', 'r', encoding='utf-8') as f:
    #     code_dic = eval(f.read())
    # return code_dic
    code_file = Path(__file__).resolve().parent.parent.parent \
        / "datasets" \
        / "city_code.json"
    with code_file.open() as f:
        code = json.load(f)
    return code


def get_html(code):
    weather_url = f'http://www.weather.com.cn/weather/{code}.shtml'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"}
    print(weather_url)
    resp = requests.get(url=weather_url, headers=header)
    resp.encoding = 'utf-8'
    return resp.text


def get_page_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    weather_info = soup.find('div', id='7d')
    seven_weather = weather_info.find('ul')
    weather_list = seven_weather.find_all('li')
    for weather in weather_list:
        print('=' * 60)
        print(weather.find('h1').get_text())
        print('天气状况：', weather.find('p', class_='wea').get_text())
        # 判断标签'p','tem'下是否有标签'span'，以此判断是否有最高温
        if weather.find('p', class_='tem').find('span'):
            temp_high = weather.find('p', class_='tem').find('span').get_text()
        else:
            temp_high = ''  # 最高温
        temp_low = weather.find('p', class_='tem').find('i').get_text()  # 最低温
        print(f'天气温度：{temp_low}/{temp_high}')
        win_list_tag = weather.find('p', class_='win').find('em').find_all('span')
        win_list = []
        for win in win_list_tag:
            win_list.append(win.get('title'))
        print('风向：', '-'.join(win_list))
        print('风力：', weather.find('p', class_='win').find('i').get_text())


def main():
    code_dic = get_city_codes()
    print('=' * 60)
    print('\t' * 5, '天气预报查询系统')
    print('=' * 60)
    city = input("请输入您要查询的城市：")
    if city in code_dic:
        html = get_html(code_dic[city]['AREAID'])
        get_page_data(html)
    else:
        print('你要查询的地方不存在')


if __name__ == '__main__':
    main()