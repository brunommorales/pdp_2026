import zipfile
from pathlib import Path

arquivo_zip = Path('../archive.zip')

destino = Path('data/raw_images/')

destino.mkdir(parents=True, exist_ok=True)

print(f"Descompactando '{arquivo_zip}' para '{destino}'...")

with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
    zip_ref.extractall(destino)

print("Arquivo descompactado com sucesso! ✅")
