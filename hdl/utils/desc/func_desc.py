FN_DESC = {
    "get_weather": """
## 函数名：get_weather
描述：只有在用户询问一个城市的天气时，调用此工具获得此城市的天气信息
触发条件：用户的问题中询问到了一个城市的天气
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
描述：只有在用户询问一个城市当前的日期或时间时，调用此工具可以获得此城市当前的日期和时间
触发条件：用户询问一个城市当前的日期或时间
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
    "web_search_text": """
## 函数名：web_search_text
描述：在用户明确说要“联网查询”回答他的问题时，调用此工具可以获得该问题联网搜索的相关内容
触发条件：用户的问题中有显式的“联网查询”字样，用户若没有提到“联网查询”，则不要调用此函数
参数：
# query_text (str): 从用户提的问题中获取，用于在网络中搜索信息
# max_results (int, optional): 搜索条目的最大数目，若用户指定了数目，则使用用户指定的数目，若用户提问中没有指定，你需要在下面的json中"max_results"这一项指定为数值3。
返回值 (str): 联网搜索到的信息
需要返回的json
{
    "function_name": "web_search_text",
    "params":
        {
            "query_text": <query from user question>,
            "max_results": <num of max results, 如果用户没有要求，这一项则指定为3>
        }
}

""",
    "execute_code(code)": """
## 函数名：execute_code,
描述：当用户明确要求执行一段代码时，调用此工具，执行这段代码，返回执行结果。
触发条件：用户要求执行一段代码
参数：
# code (str): 用户要求执行的代码
返回值 (str): 执行结果
需要返回的json
{
    "function_name": "execute_code",
    "params":
        {
            "code": <code>
        }
}
""",
    "default": None
}