import pickle
from pathlib import Path
import re

def save_versioned_pickle_file(obj, object_name, folder='.'):
    """
    Saves `obj` into a versioned Pickle file in `folder`.
    
    If no files named {object_name}_X.pickle exist, creates {object_name}_1.pickle.
    
    Otherwise the function creates {object_name}_{max_version+1}.pickle.
    """
    folder_path = Path(folder)
    # Pattern to match files like object_name_3.pickle
    pattern = re.compile(rf"^{re.escape(object_name)}_(\d+)\.pickle$")
    max_version = 0
    for file in folder_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
    new_version = max_version + 1
    # Write object to Pickle file
    print(f"{folder}/{object_name}_{new_version}.pickle")
    with open(f"{folder}/{object_name}_{new_version}.pickle", 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_dataset_from_file(filename):
    data_file = open(filename, 'rb')
    model_data = pickle.load(data_file)
    return model_data