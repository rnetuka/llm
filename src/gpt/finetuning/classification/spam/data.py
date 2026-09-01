from pathlib import Path
from urllib import request
from zipfile import ZipFile


resource_directory = Path('..') / 'resources' / 'finetuning'


def download_sms_spam_data() -> Path:
    data_file_path = resource_directory / 'SMSSpamCollection.tsv'

    if data_file_path.exists():
        return data_file_path

    url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'

    if not resource_directory.exists():
        resource_directory.mkdir()

    zip_path = resource_directory / 'sms_spam_collection.zip'

    with request.urlopen(url) as response:
        with open(zip_path, 'wb') as file:
            file.write(response.read())

    with ZipFile(zip_path, 'r') as zip:
        zip.extractall(resource_directory)

    # rename the unzipped content
    (resource_directory / 'SMSSpamCollection').rename(data_file_path)

    # clen up the unused files
    (resource_directory / 'sms_spam_collection.zip').unlink()
    (resource_directory / 'readme').unlink()
    return data_file_path
