# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/network/proxy.py
# 说明：网络与代理工具
import requests


def get_proxies(
    pool_server='http://172.20.0.9:5010',
    https=False
):
    """从代理池服务取一个可用代理并组装成 requests 的 proxies 字典：向 pool_server 的 /get/ 取 http 代理、/get/?type=https 取 https 代理（两次独立 GET，响应需是含 proxy 字段的 JSON），
    https=True 时两个键都用 https 代理地址，否则都用 http 代理地址；键值统一按 http 键配 http:// 前缀、https 键配 https:// 前缀拼出，
    返回 requests 可直接使用的 {'http': ..., 'https': ...} 字典；服务不可达或响应里没有 proxy 字段时在 json()/拼串处抛异常，本函数只做两次 GET、不缓存也不校验代理可用性。"""
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