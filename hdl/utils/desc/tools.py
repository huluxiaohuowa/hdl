TOOL_DICT = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "仅在用户明确询问天气、气候、气温等信息时，获取指定城市的天气情况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name e.g. Bogotá"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
}