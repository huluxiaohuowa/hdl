TOOL_DESC = {

    "get_weather":{
        "desc": """
## 函数名：get_weather
描述：用户询问一个城市的天气时，调用此工具获得此城市的天气信息
参数：
# city (str): 城市名
返回值 (str)：天气信息
        """,
        "json": """
需要返回的json
{
    "function_name": "get_weather",
    "params":
        {
            "city": <city_name>
        }
}

        """
    },

    "get_datetime_by_cityname": {
        "desc": """
## 函数名：get_datetime_by_cityname
描述：用户询问一个城市当前的日期或时间时，调用此工具可以获得此城市当前的日期和时间
参数：
# city (str): 城市名
返回值 (str):这个城市当前所在时区的日期和时间
        """,
        "json": """
需要给出的json
{
    "function_name": "get_datetime_by_cityname",
    "params":
        {
            "city": <city_name>
        }
}
        """
    },

    "execute_code": {
        "desc": """
## 函数名：execute_code,
描述：当用户明确要求执行一段代码时，调用此工具，执行这段代码，返回执行结果。
参数：
# code (str): 用户要求执行的代码
返回值 (str): 执行结果
        """,
        "json": """
需要返回的json
{
    "function_name": "execute_code",
    "params":
        {
            "code": <code>
        }
}
        """
    },

    "calculate": {
        "desc": """
## 函数名：calculate，
描述：当需要计算一个直接的在 python 中可交互的数据学算式，调用此工具，计算这个表达式，返回计算结果。
参数：
# expression (str): 用户要求的数学表达式
返回值 (str): 计算结果
        """,
        "json": """
需要返回的json
{
    "function_name": "calculate",
    "params":
        {
            "expression": <expression>
        }
}
        """
    },

    "fetch_baidu_results": {
        "desc": """
## 函数名：fetch_baidu_results,
描述：在用户提的问题需要联网搜索才能得到结果或者问题中有联网搜索的需求时，调用此工具，返回查询结果。
参数：
# query (str): 用户要求的查询内容
# max_n_links (int, optional): 搜索条目的最大数目，若用户指定了数目，则使用用户指定的数目，若用户提问中没有指定，你需要在下面的json中"max_n_links"这一项指定为数值3。
返回值 (str): 联网搜索到的信息
        """,
        "json": """
需要返回的json
{
    "function_name": "fetch_baidu_results",
        "params":
        {
            "query": <query from user question>,
            "max_n_links": <num of max results, 如果用户没有要求，这一项则指定为3>
        }
}
        """
    },

    "wolfram_alpha_calculate": {
        "desc": """
## 函数名：wolfram_alpha_calculate
描述：当用户要求计算一个明确用数学语言描述的表达式时，调用此工具，返回计算结果。注意不能描述物理、化学、应用题等非数学表达式。
参数：
# query (str): 用户要求的数学表达式，需要转换为英文，比如"integrate sin(x)"
返回值 (str): 计算结果
        """,
        "json": """
需要返回的json
{
    "function_name": "wolfram_alpha_calculate",
    "params":
        {
            "query": <expression>
        }
}
        """
    },

    "count_character_occurrences": """
## 函数名：count_character_occurrences
描述：当用户要求计算一段文本中某个字符出现的次数时，调用此工具，返回计算结果。
参数：
# text (str): 输入的单词或句子
# char (str): 要统计的字符
返回值 (str): 字符在文本中出现的次数
""",
    "json": """
需要返回的json
{
    "function_name": "count_character_occurrences",
    "params":
        {
            "text": <str>,
            "char": <str>
        }
}
""",

    "default": None

}