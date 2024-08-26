import psycopg
from psycopg import sql


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


def get_item_by_idx(
    idx: int,
    info_file: str,
    by: str = id,
    table: str = 'reaction_id'
):

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