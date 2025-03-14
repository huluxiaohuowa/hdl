# import os
# import pty
# import subprocess
# # import sys

# def run_ollama_cmd(cmd: str):
#     command = cmd.split(" ")
#     # 定义要执行的命令
#     # command = ["ollama", "create", ollama_model_name, "-f", ollama_model_file]
#     # 创建伪终端
#     master, slave = pty.openpty()

#     # 运行命令
#     process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, text=True, close_fds=True)

#     # 关闭 slave 端，让 master 端可以读取
#     os.close(slave)

#     # 处理动态进度条，避免重复输出
#     seen_lines = set()
#     while True:
#         try:
#             output = os.read(master, 1024).decode()
#             if not output:
#                 break

#             # 过滤重复的进度信息
#             for line in output.split("\n"):
#                 if line.strip() and line not in seen_lines:
#                     seen_lines.add(line)
#                     print(line, flush=True)  # 逐行打印
#         except OSError:
#             break

#     # 等待进程结束
#     process.wait()

import subprocess

def run_cmd(cmd: str):
    command = cmd.split(" ")

    # 直接执行命令，让 stdout 和 stderr 直接流式输出到终端
    process = subprocess.Popen(command, stdout=None, stderr=None)

    # 等待进程执行完
    process.wait()

run_cmd("ollama push ictrek/qwen2.5:14b32ft")