import psycopg


def connect_by_infofile(info_file: str) -> psycopg.Connection:
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