import subprocess
import sys
import os
import math

def execute_code(code):
    """
    Executes the given Python code string in a subprocess.

    This function uses the subprocess to execute Python code strings, limiting the execution time to prevent infinite loops,
    and captures the execution output and errors for returning results.

    Parameters:
    code (str): The Python code string to be executed.

    Returns:
    str: The output result of executing the code or an error message.
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
    Calculates the result of a mathematical expression.

    This function evaluates a mathematical expression string passed as an argument and returns the result.
    In order to prevent security risks, the use of built-in functions is restricted during evaluation,
    allowing only the 'math' module to be accessed. If an exception occurs during evaluation,
    it returns the corresponding error message.

    Parameters:
    expression (str): The mathematical expression to calculate.

    Returns:
    The result of the calculation or an error message if an exception occurs.
    """
    try:
        # Evaluate the expression with restricted built-in functions and access to the math module
        return eval(expression, {"__builtins__": None}, {"math": math})
    except Exception as e:
        # Return the error message if an exception occurs
        return f"Error: {str(e)}"