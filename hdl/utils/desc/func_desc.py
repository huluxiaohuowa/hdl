FN_DESC = {
    "get_weather": """
## 函数名：get_weather
描述：在用户询问一个城市的天气时，调用此工具获得此城市的天气信息
参数：
# city (str): 城市名
返回值 (str)：天气信息
需要返回的json
{
    "function_name": "get_weather",
    "params":
        {
            "city": <city_name>
        }
}

""",
    "get_datetime_by_cityname": """
## 函数名：get_datetime_by_cityname
描述：在用户询问一个城市当前的日期或时间时，调用此工具可以获得此城市当前的日期和时间
参数：
# city (str): 城市名
返回值 (str):这个城市当前所在时区的日期和时间
需要返回的json
{
    "function_name": "get_datetime_by_cityname",
    "params":
        {
            "city": <city_name>
        }
}

""",
    "default": None
}