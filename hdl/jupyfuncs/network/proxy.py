# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/network/proxy.py
# 说明：网络与代理工具
import requests


def get_proxies(
    pool_server='http://172.20.0.9:5010',
    https=False
):
    http_proxy = requests.get(f"{pool_server}/get/").json().get("proxy")
    https_proxy = requests.get(f"{pool_server}/get/?type=https").json().get("proxy")
    if https:
        proxy_dict = {
            'http': f'http://{https_proxy}',
            'https': f'https://{https_proxy}'
        }
    else:
        proxy_dict = {
            'http': f'http://{http_proxy}',
            'https': f'https://{http_proxy}'
        }
    return proxy_dict