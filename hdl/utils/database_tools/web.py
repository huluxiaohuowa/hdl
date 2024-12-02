import os

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
    from duckduckgo_search import DDGS
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
    Fetches search results from Baidu for a given query and retrieves the text content of the top results.
    Args:
        query (str): The search query to be sent to Baidu.
        max_n_links (int, optional): The maximum number of search result links to fetch. Defaults to 3.
    Returns:
        str: The concatenated text content of the fetched web pages.
    Raises:
        requests.RequestException: If there is an issue with the HTTP request.
    """
    try:
        max_n_links = int(max_n_links)
    except Exception as e:
        print(e)
        max_n_links = 3
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


def wolfram_alpha_calculate(query):
    """
    Sends a query to the Wolfram Alpha API and returns the result.
    Args:
        query (str): The query string to be sent to Wolfram Alpha.
    Returns:
        str: The result of the query in plaintext format, or an error message if the query was unsuccessful or if an error occurred.
    Raises:
        requests.Timeout: If the request to Wolfram Alpha times out.
        Exception: If any other error occurs during the request.
    """
    # Get the Wolfram Alpha App ID from environment variables
    app_id = os.getenv('WOLFRAM_APP_ID', None)
    if not app_id:
        return "Error: Wolfram Alpha App ID is not set in environment variables."

    # Define the API endpoint for Wolfram Alpha
    url = 'https://api.wolframalpha.com/v2/query'
    # Prepare the request parameters
    params = {
        'appid': app_id,
        'input': query,
        'output': 'json',
    }

    try:
        # Send a GET request to Wolfram Alpha
        response = requests.get(url, params=params, timeout=20)
        # Parse the returned data as JSON
        data = response.json()

        # Check if the query was successful
        if data['queryresult']['success']:
            result = ''
            # Iterate through the returned data to extract and accumulate the plaintext results
            for pod in data['queryresult']['pods']:
                for subpod in pod.get('subpods', []):
                    plaintext = subpod.get('plaintext')
                    if plaintext:
                        result += f"{plaintext}\n"
            # Return the accumulated result, or a message if no plaintext result is available
            return result.strip() if result else "No plaintext result available."
        else:
            return "No results found for the query."
    except requests.Timeout:
        return "Error: The request to Wolfram Alpha timed out."
    except Exception as e:
        return f"An error occurred: {str(e)}"