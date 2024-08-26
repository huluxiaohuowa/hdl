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