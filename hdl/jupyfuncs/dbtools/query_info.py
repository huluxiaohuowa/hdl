import re

import cirpy
import pubchempy as pcp
import molvs as mv
from psycopg import sql

from .pg import connect_by_infofile


def query_from_cir(query_name: str):
    smiles = None
    # cas_list = []
    # name_list = [host=172.20.0.5 dbname=pistachio port=5432 user=postgres password=woshipostgres]

    cas_list = cirpy.resolve(query_name, 'cas')
    if cas_list is None or not cas_list:
        cas_list = []
    if isinstance(cas_list, str):
        cas_list = [cas_list]

    name_list = cirpy.resolve(query_name, 'names')
    if name_list is None or not name_list:
        name_list = []
    if isinstance(name_list, str):
        name_list = [name_list]

    smiles = cirpy.resolve(query_name, 'smiles')
    try:
        smiles = mv.standardize_smiles(smiles)
    except Exception as e:
        print(e)

    return smiles, cas_list, name_list


def query_from_pubchem(query_name: str):
    results = pcp.get_compounds(query_name, 'name')
    smiles = None
    name_list = set()
    cas_list = set()

    if any(results):
        try:
            smiles = mv.standardize_smiles(results[0].canonical_smiles)
        except Exception as e:
            smiles = results[0].canonical_smiles
            print(smiles)
            print(e)
        for compound in results:
            name_list.update(set(compound.synonyms))
            for syn in compound.synonyms:
                match = re.match('(\d{2,7}-\d\d-\d)', syn)
                if match:
                    cas_list.add(match.group(1))
        
        cas_list = list(cas_list)
        name_list = list(name_list)

    return smiles, cas_list, name_list


def query_a_compound(
    query_name: str,
    connect_info: str,
    by: str = 'name',
    log_file: str = './err.log'
):
    fei = None
    found = False  

    if by != 'smiles':
        query_name = query_name.lower()
    
    by = 'name'
    table = by + '_maps'
    # query_name = 'adipic acid'
    query = sql.SQL(
        "select fei from {table} where {by} = %s"
    ).format(
        table=sql.Identifier(table),
        by=sql.Identifier(by)
    )
    conn = connect_by_infofile(connect_info)
    
    cur = conn.execute(query, [query_name]).fetchone()

    if cur is not None:
        fei = cur[0]
        found = True
        return fei
 
    if not found:
        try:
            smiles, cas_list, name_list = query_from_pubchem(query_name) 
        except Exception as e:
            print(e)
            smiles, cas_list, name_list = None, [], []
        if smiles is not None:
            found = True
        else:
            try:
                smiles, cas_list, name_list = query_from_cir(query_name)
            except Exception as e:
                print(e)
                smiles, cas_list, name_list = None, [], []
            if smiles is not None:
                found = True
    
    if not found:
        with open(log_file, 'a') as f:
            f.write(query_name)
            f.write('\n')
        return
        # raise ValueError('给的啥破玩意儿查都查不着！')
    else:
        query_compound = sql.SQL(
            "select fei from compounds where smiles = %s"
        )
        cur = conn.execute(query_compound, [smiles]).fetchone()
        if cur is not None:
            fei = cur[0]
        elif any(cas_list):
            fei = cas_list[0]
            insert_compounds_sql = sql.SQL(
                "INSERT INTO compounds (fei, smiles) VALUES (%s, %s) ON CONFLICT (fei) DO NOTHING"
            )
            conn.execute(insert_compounds_sql, [fei, smiles])
        for cas in cas_list:
            insert_cas_map_sql = sql.SQL(
                "INSERT INTO cas_maps (fei, cas) VALUES (%s, %s) ON CONFLICT (cas) DO NOTHING"
            )
            try:
                conn.execute(insert_cas_map_sql, [fei, cas])
            except Exception as e:
                print(e)
        for name in name_list:
            insert_name_map_sql = sql.SQL(
                "INSERT INTO name_maps (fei, name) VALUES (%s, %s) ON CONFLICT (fei, name) DO NOTHING"
                # "INSERT INTO name_maps (fei, name) VALUES (%s, %s)"
            ) 
            try:
                conn.execute(insert_name_map_sql, [fei, name.lower()])
            except Exception as e:
                print(e)

    conn.commit()
    conn.close()

    return fei
