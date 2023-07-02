import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_url(file_name, url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        absolute_url = urljoin(url, link.get('href'))
        file_url = absolute_url.split("/")[-1]

        if file_name in file_url:
            print(f"Downloading {file_url} from {absolute_url}")

            file_response = requests.get(absolute_url, stream=True)

            download_path = os.path.join(os.getcwd(), file_url)
            with open(download_path, 'wb') as file:
                for chunk in file_response.iter_content(chunk_size=128):
                    file.write(chunk)

            print(f"(file_url) downloaded successfully to {download_path}")

for num in range(1, 23):
    file_name = f"cs188-fa22-note{num}.pdf"
    download_url(file_name, "https://inst.eecs.berkeley.edu/~cs188/fa22/")
print("You successfully downloaded all files!")
