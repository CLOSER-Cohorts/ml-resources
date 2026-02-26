import pickle
from pathlib import Path
import re

def save_versioned_pickle_file(obj, object_name):
    """
    Saves `obj` into a versioned Pickle file in the `..\data` folder.
    
    If no files named {object_name}_X.pickle exist, creates {object_name}_1.pickle.
    
    Otherwise the function creates {object_name}_{max_version+1}.pickle.
    """
    folder_path = Path("../data")
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
    print(f"../data/{object_name}_{new_version}.pickle")
    with open(f"../data/{object_name}_{new_version}.pickle", 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_for_ml=[1,2,3,4,5,6]

save_versioned_pickle_file(data_for_ml, 'data_for_ml')    
