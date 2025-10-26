import requests
from tqdm import tqdm
import os

def download_file(url: str, destination: str) -> None:
    """
    Downloads a file from the given URL with a progress bar.

    Args:
        url (str): URL of the file to download.
        destination (str): Path to save the downloaded file.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))

    print(f"\nDownload complete: {destination}")
