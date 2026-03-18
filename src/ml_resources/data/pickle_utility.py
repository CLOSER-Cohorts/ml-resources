import pickle
from pathlib import Path
import re

def save_versioned_pickle_file(obj, object_name, folder='.', comments="New version of file."):
    """
    Saves `obj` into a versioned Pickle file in `folder`.
    
    If no files named {object_name}_X.pickle exist, creates {object_name}_1.pickle.
    
    Otherwise the function creates {object_name}_{max_version+1}.pickle.
    """
    folder_path = Path(f"{folder}/{object_name}/")
    #file_path.parent.mkdir(parents=True, exist_ok=True)
    folder_path.mkdir(parents=True, exist_ok=True)
    #folder_path = Path(folder)
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
    file_path = Path(f"{folder}/{object_name}/{object_name}_{new_version}.pickle")
    print(file_path)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    generate_data_versioning_document(f"{folder}/{object_name}", object_name, new_version, comments)

def generate_data_versioning_document(data_documentation_folder,
    dataset_name,
    version_number,
    version_comments):
    with open(f"{data_documentation_folder}/{dataset_name}_notes_{version_number}.txt", "w") as f:
        f.write(version_comments)

def read_dataset_from_file(filename):
    data_file = open(filename, 'rb')
    model_data = pickle.load(data_file)
    return model_data