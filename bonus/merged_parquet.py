import os
import awkward as ak

# Define the directory path
directory = '/eos/user/s/shsong/HiggsDNA/GJet16post'

# Get a list of all subdirectories
subdirectories = [x for x in os.listdir(directory)]

# Initialize an empty list to store the arrays
arrays = []

# Iterate over each subdirectory
for subdir in subdirectories:
    # print('subdir', subdir)
    # if subdir is not dir continue
    subdir_path = os.path.join(directory, subdir)
    print('subdir_path:', subdir_path)
    if not os.path.isdir(subdir_path):
        continue
    # Get a list of all parquet files in the subdirectory
    parquet_files = [file for file in os.listdir(subdir_path) if file.endswith('.parquet')]
    print("parquet_files:",parquet_files)
    # Iterate over each parquet file
    for file in parquet_files:
        if "GJets" in os.path.join(subdir_path, file):
            print('Reading file:', os.path.join(subdir_path, file))
            # Read the parquet file into an array
            array = ak.from_parquet(os.path.join(subdir_path, file))
            # Append the array to the list
            arrays.append(array)
        else:
            continue
# Concatenate all arrays into a single array
merged_array = ak.concatenate(arrays)

# Save the merged array as a parquet file
ak.to_parquet(merged_array, '/eos/user/s/shsong/HiggsDNA/GJet16post/merged_output.parquet')