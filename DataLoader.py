import os
import zipfile
import hashlib
import requests
from tqdm.auto import tqdm

class ADE20KDownloader:
    def __init__(self, path):
        self.path = path
        self._AUG_DOWNLOAD_URLS = [
            ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
            ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),
        ]

    def download_file(self, url, output_path, sha1_hash=None):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if sha1_hash:
            if not self.validate_file(output_path, sha1_hash):
                raise Exception("Downloaded file hash does not match expected hash.")
        
        return output_path

    def validate_file(self, file_path, sha1_hash):
        sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                sha1.update(data)
        calculated_hash = sha1.hexdigest()
        return calculated_hash == sha1_hash

    def download_ade(self, overwrite=False):
        """Download ADE20K"""
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        download_dir = os.path.join(self.path, 'downloads')
        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
        for url, checksum in self._AUG_DOWNLOAD_URLS:
            filename = os.path.join(download_dir, os.path.basename(url))
            if not os.path.exists(filename) or overwrite:
                self.download_file(url, filename, sha1_hash=checksum)
                # extract
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(path=self.path)

