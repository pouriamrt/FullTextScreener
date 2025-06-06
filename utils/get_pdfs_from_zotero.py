import pandas as pd
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

output_dir = Path("data/papers")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    # Exported csv file from Zotero
    df = pd.read_csv("data/Exported Items.csv")
    df = df[df['File Attachments'].notna()]
    
    df = df[df['File Attachments'].str.endswith('.pdf')]
    
    df['File Attachments'] = df['File Attachments'].str.replace(';', '').str.strip()
    files_list = df['File Attachments'].tolist()
    
    for file in tqdm(files_list):
        source_path = Path(file)
        dest_path = output_dir / source_path.name
        
        if not source_path.exists():
            print(f"Warning: Source file {file} does not exist")
            continue
            
        if dest_path.exists():
            print(f"Warning: File {dest_path.name} already exists in destination, skipping")
            continue
            
        try:
            subprocess.run(['cp', str(source_path), str(output_dir)], check=True)
            # print(f"Successfully copied {source_path.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error copying {source_path.name}: {str(e)}")
            
except FileNotFoundError:
    print("Error: Could not find 'data/Exported Items.csv'")
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
