from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz

def get_datetime_by_cityname(city):
    """Get current date and time of a city based on its name.

    Args:
        city_name (str): The name of the city to get date and time for.

    Returns:
        str: Formatted date and time string in the format '%Y年%m月%d日 %H:%M:%S'.
    """
    # 使用 Nominatim API 通过城市名称获取地理坐标
    geolocator = Nominatim(user_agent="city_time_locator")
    location = geolocator.geocode(city)

    if not location:
        return f"无法找到城市 '{city}'，请检查输入的城市名称。"

    # 使用地理坐标获取时区
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lng=location.longitude, lat=location.latitude)

    if not timezone_name:
        return f"无法找到城市 '{city}' 的时区。"

    # 获取当前时间
    timezone = pytz.timezone(timezone_name)
    city_time = datetime.now(timezone)

    # 返回格式化的日期时间字符串
    time_str = city_time.strftime('%Y年%m月%d日 %H:%M:%S')
    time_str = f"{city}现在的时间为" + time_str
    return time_str