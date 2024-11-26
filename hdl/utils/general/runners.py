import subprocess
import sys
import os
import math

def execute_code(code):
    """
    Executes the given Python code string in a subprocess and returns the execution result.

    Args:
        code (str): The Python code to be executed, provided as a string.

    Returns:
        str: The output result of the code execution. If there is an error during execution,
             it returns an error message describing the error.
    """
    try:
        # Execute the code in a subprocess for safety
        print(code)
        python_path = sys.executable  # 获取当前 Python 解释器路径
        result = subprocess.run(
            [
                python_path,  # 使用当前 conda 环境中的 Python
                '-c',
                code
            ],
            capture_output=True,
            text=True,
            timeout=5,
            env={"PYTHONPATH": os.getcwd()}
        )
        # Return the execution result based on the subprocess return code
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        # Return a timeout error message if the code execution exceeds the limit
        return "Error: Code execution timed out"
    except Exception as e:
        # Return the exception information for other exceptions
        return f"Error: {str(e)}"


def calculate(expression):
    """
    Safely evaluates a mathematical expression.

    This function receives a string containing a mathematical expression and evaluates it safely, allowing access only to the math module.
    If an error occurs during evaluation, such as a syntax error or an undefined function, the function catches the exception and returns an error message.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        The result of the expression evaluation or an error message if an exception occurs.
    """
    try:
        # Evaluate the expression with restricted built-in functions and access to the math module
        return eval(expression, {"__builtins__": None}, {"math": math})
    except Exception as e:
        # Return the error message if an exception occurs
        return f"Error: {str(e)}"

def count_character_occurrences(text, char):
    """
    统计文本中某个字符出现的次数

    Args:
        text (str): 输入的单词或句子
        char (str): 要统计的字符（仅支持单个字符）

    Returns:
        int: 字符在文本中出现的次数
    """
    if not isinstance(text, str):
        raise ValueError("输入的文本必须是字符串类型")
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("要统计的字符必须是一个单字符的字符串")
    text = text.lower()
    char = char.lower()

    return f"{text} 中 {char} 共出现了 {text.count(char)} 次。"