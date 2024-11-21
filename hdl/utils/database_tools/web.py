from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

def web_search_text(
    query_text: str,
    max_results=3,
):
    """Searches the web for text related to the given query.

    Args:
        query_text (str): The text to search for.
        max_results (int, optional): The maximum number of results to retrieve. Defaults to 3.

    Returns:
        str: Text retrieved from the web search results.
    """
    if max_results < 3:
        max_results = 3
    elif max_results > 5:
        max_results = 5
    result_str = "联网搜索到的信息如下：\n"
    try:
        results = DDGS().text(
            query_text,
            max_results=max_results,
            backend="html"
        )
    except Exception as e:
        return f"{str(e)}: 未搜索（获取）到相关内容，可过一段时间再次尝试。"
    for result in results:
        if "wikipedia" not in result['href']:
            print(f"Getting info from {result['href']}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
            }
            try:
                # 发送请求
                response = requests.get(result['href'], headers=headers, timeout=10)

                # 检查请求状态码
                if response.status_code == 200:
                    # 解析网页内容
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()

                    # 删除空行
                    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

                    result_str += cleaned_text
                    result_str += "\n"
            except Exception as e:
                print(f"{str(e)}: 从{result['href']}未搜索（获取）到相关内容。")
    return result_str


def fetch_baidu_results(query, max_n_links=3):
    """
    模拟百度搜索，提取前三个搜索结果的网页文字内容并拼接返回。

    :param query: str 要搜索的文本
    :return: str 提取的网页内容拼接字符串
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # 百度搜索 URL
    search_url = 'https://www.baidu.com/s'
    params = {'wd': query}

    # 发送搜索请求
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取前三个搜索结果链接
    links = []
    for link_tag in soup.select('.t a')[:max_n_links]:
        link = link_tag.get('href')
        if link:
            links.append(link)

    # 抓取每个链接的网页内容
    text_content = []
    for link in links:
        try:
            page_response = requests.get(link, headers=headers, timeout=10)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            text = page_soup.get_text(separator='\n', strip=True)
            text_content.append(text)
        except requests.RequestException as e:
            print(f"Failed to fetch {link}: {e}")

    # 返回拼接的文本内容
    return '\n'.join(text_content)