import subprocess
import sys
import os

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