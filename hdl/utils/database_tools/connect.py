import psycopg
import redis

def connect_by_infofile(info_file: str):
    """Create a postgres connection

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
    host: str,
    port: int
):
    import redis
    client = redis.Redis(
        host=host,
        port=port,
        decode_responses=True
    )
    res = client.ping()
    print(res)
    return client