import os

def make_folder(output_dir, folder_name):
    output_path = os.path.join(output_dir, folder_name)

    # Check if the base folder already exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        return output_path
    else:
        # Try to find a unique folder name with a suffix
        suffix = 1
        while True:
            new_output_path = os.path.join(output_dir, f"{folder_name}{suffix}")
            if not os.path.exists(new_output_path):
                os.makedirs(new_output_path)
                return new_output_path
            suffix += 1