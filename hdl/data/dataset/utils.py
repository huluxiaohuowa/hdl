import typing as t
import csv



def read_smiles(
    data_path: str,
    file_type: str = 'smi',
    smi_col_names: t.List = [],
    y_col_name: str = 'None',
):
    smiles_data = []
    if file_type == 'smi':
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                smiles = row[-1]
                smiles_data.append(smiles)
    elif file_type == 'csv' and any(smi_col_names):
        # for _ in smi_col_names:
        #     smiles_data.append([])
        with open(data_path, 'r') as theFile:
            reader = csv.DictReader(theFile)
            for line in reader:
                # line is { 'workers': 'w0', 'constant': 7.334, 'age': -1.406, ... }
                # e.g. print( line[ 'workers' ] ) yields 'w0'
                smiles_data_i = [line[i] for i in smi_col_names]
                if y_col_name is not None:
                    smiles_data_i.append(line[y_col_name])
                smiles_data.append(smiles_data_i)
    return smiles_data