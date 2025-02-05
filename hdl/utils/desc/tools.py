TOOL_DICT = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
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