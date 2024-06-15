import os

def make_folder(output_dir, base_folder_name):
    # Check if the base folder already exists
    if not os.path.exists(os.path.join(output_dir)):
        output_path = os.path.join(output_dir, base_folder_name)
        os.makedirs(output_path)
        return output_path
    else:
        # Try to find a unique folder name with a suffix
        suffix = 1
        while True:
            new_outupt_dir = f"{output_dir}{suffix}"
            if not os.path.exists(new_outupt_dir):
                output_path = os.path.join(new_outupt_dir, base_folder_name)
                os.makedirs(output_path)
                return output_path
            suffix += 1