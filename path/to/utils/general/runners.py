def run_cmd(command: str):
    """
    Executes a command in a subprocess and returns the execution result.

    Args:
        command (str): A string representing the command to execute. The parts of the command should be separated by spaces.

    Returns:
        str: The return message of the command execution. If there is an error during execution,
             it returns an error message describing the error.

    Raises:
        ValueError: If the input command is not a string.
    """
    if not isinstance(command, str):
        raise ValueError("Command must be a string")

    command = command.split(" ")

    # 直接执行命令，让 stdout 和 stderr 直接流式输出到终端
    process = subprocess.Popen(command, stdout=None, stderr=None)

    # 等待进程执行完
    process.wait()
