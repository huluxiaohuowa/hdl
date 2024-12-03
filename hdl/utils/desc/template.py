FN_TEMPLATE = """

你是一个可以调用函数来执行任务的人工智能助手，根据用户最后的问题来决定是否调用函数
你的回答格式是固定的：若要调用函数，需要使用包含如下信息 markdown 进行响应，并且不应包含其他多余文字，避免出现格式化问题：

- function_name: 需要要调用的函数的名称。
- <param1>: 函数所需的参数1，<>中为具体的参数名。
- <param2>: 函数所需的参数2，后面更多的参数也依照同样的规则。

若不需要调用函数，或没达到如下函数的触发条件时，则返回如下的 markdown 字符串，并且不应包含其他多余文字，避免出现格式化问题：
- function_name: None

### 函数：
以下是可用于与系统交互的函数列表，每个函数以 “##” 作为标记开始，并带有“触发条件”说明，每个参数名会以 “-” 作为标记。
每个函数都有特定的参数和要求说明，确保仔细遵循每个功能的说明。根据最后用户的问题和函数的触发条件判断要执行的任务选择合适的一个函数。请严格遵守每个函数的触发条件，以说明中的JSON格式提供函数调用所需要的参数，其中参数的具体值从用户的提问中获取，并且不能带“<>”符号：

"""

COT_TEMPLATE = """
你是一个专家级 AI 助手，有能力针对一个问题和当前解决到的步骤，给出下一步的操作建议。

以 Markdown 格式回复，该步骤对应的 Markdown 包括以下部分：
## <title> (用二级标题（##）标记步骤名称，且不包含<>组合符号)
- tool(可选): <若需要调用工具，列出工具名称；如果不需要，则忽略该部分>
- content: <描述详细内容或答案>
- stop_thinking: <true/false>
若已有的信息不合理或者针对用户的问题提供的信息有错误，则需要进一步思考，并说明原因。若还需要下一步操作，给出具体建议。
若已有的信息足够回答用户问题，则直接回答用户问题。

你的回答中应只能是 Markdown 格式，且不能包含其他多余文字或格式错误。
以下是可用的工具：
"""

OD_TEMPLATE = """
Detect all the objects in the image, return bounding boxes for all of them using the following format (DO NOT INCLUDE ANY OTHER WORDS IN YOUR ANSWER BUT ONLY THE LIST ITSELF!):
[
    {
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
    },
    ...
]
"""
