# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/database_tools/connect.py
# 说明：数据库连接与网页数据工具
import psycopg
import redis

def connect_by_infofile(info_file: str):
    """Create a postgres connection
    把连接信息文件的首行整串作为连接参数交给 psycopg.connect（文件按 libpq 的 key=value 风格存放主机、端口、库名与账号口令，具体值不写进注释），
    即每次调用都新建一条 PostgreSQL 连接、不会复用；返回的连接不由本函数关闭，需调用方 commit 后 close。

    Args:
        info_file (str):
            the path of the connection info like
            host=127.0.0.1 dbname=dbname port=5432 user=postgres password=lala

    Returns:
        psycopg.Connection:
            the connection instance should be closed after committing.
    """
    conn = psycopg.connect(
        open(info_file).readline()
    )
    return conn
# with psycopg.connect(
#     open('./conn.info').readline()
# ) as conn:
#     cur = conn.execute('select * from name_maps;')
#     cur.fetchone()
#     for record in cur:
#         print(record)
#     conn.commit()
#     conn.close()

def conn_redis(
    **redis_args
):
    """按关键字参数新建 Redis 客户端并顺带探活：redis_args 原样透传给 redis.Redis（host、port、db、password 等连接项都在其中，取值不写进注释），
    构造后立刻 client.ping() 并把返回布尔值打印到标准输出；返回值是已建立的 redis.Redis 实例（连接由该对象持有，函数内不关闭），参数不合法或服务不可达时在 Redis 构造/ping 阶段抛异常。"""
    import redis
    client = redis.Redis(
        **redis_args
    )
    res = client.ping()
    print(res)
    return client