# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/jupyfuncs/dbtools/pg.py
# 说明：数据库查询工具
import psycopg
from psycopg import sql


def connect_by_infofile(info_file: str) -> psycopg.Connection:
    """Create a postgres connection
    把连接信息文件的首行整串交给 psycopg.connect 建立 PostgreSQL 连接（该文件按 libpq 的 key=value 形式存放主机、端口、库名与账号口令，取值不在注释中复现）；只读文件不写，返回的连接需由调用方 commit 后 close。

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


def get_item_by_idx(
    idx: int,
    info_file: str,
    by: str = id,
    table: str = 'reaction_id'
):
    """按 idx 到 PostgreSQL 查一条 reaction_id：先 connect_by_infofile(info_file) 取连接（本函数不 commit 也不 close），
    再用 psycopg.sql 把 table 与 by 拼成标识符生成 select reaction_id from <table> where <by> = %s，比较值取 str(idx)（即十进制字符串形式的 id 列值）；
    返回 execute().fetchone() 的首列，查不到记录时 fetchone 为 None、下标取值抛 TypeError。形参 by 标注为 str 但默认值是内置函数 id，
    不显式传列名时无法作为标识符使用，因此调用方必须传入用于等值匹配的列名。"""

    conn = connect_by_infofile(
        info_file
    )

    query_name = str(idx)
    query = sql.SQL(
        "select reaction_id from {table} where {by} = %s"
    ).format(
        table=sql.Identifier(table),
        by=sql.Identifier(by)
    )
    cur = conn.execute(query, [query_name]).fetchone()
    return cur[0]