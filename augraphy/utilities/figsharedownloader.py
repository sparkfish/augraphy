import json
import os
import random
import shutil
from urllib.request import urlretrieve

import requests
from requests.exceptions import HTTPError


class FigshareDownloader:
    """Makes HTTP requests for images on Figshare"""

    def __init__(self, directory="figshare/"):
        self.save_dir = os.path.join(os.getcwd(), directory)

    def make_files_url(self, article_id):
        """Form the full URL for requests"""
        return f"https://api.figshare.com/v2/articles/{article_id}/files"

    def make_save_dir(self):
        # Don't throw errors if we download stuff multiple times
        os.makedirs(self.save_dir, exist_ok=True)

    def send_request(self, url, headers):
        """Request Figshare data

        :param url: request endpoint
        :type url: string
        :param headers: header info for request
        :type headers: dictionary
        :param data: Figshare article data
        :type data: dictionary
        :param binary: True if downloading images
        :type binary: boolean, optional
        """

        response = requests.request("GET", url, headers=headers, data=None)

        try:
            response.raise_for_status()

            try:
                response_data = json.loads(response.text)

            except ValueError:
                response_data = response.content

        except HTTPError as error:
            print(f"HTTP Error: {error}")
            print(f"Response Body:\n {response.text}")
            raise

        return response_data

    def list_article_files(self, article_id):
        """Get a dictionary of files from Figshare.

        :param article_id: ID of the Figshare article
        :type article_id: string
        """

        request_url = self.make_files_url(article_id)
        request_header = {"Content-Type": "application/json"}
        response = self.send_request(request_url, headers=request_header)
        return response

    def download_file_by_id(self, file_id, file_name=None):
        """Download a single file using its unique identifier,
        and optionally rename it.

        :param file_id: ID of the Figshare file
        :type id: string
        """

        # Make ./figshare/ if not available
        self.make_save_dir()

        local_file, headers = urlretrieve(
            f"https://figshare.com/ndownloader/files/{file_id}",
        )

        if file_name is not None:
            shutil.move(local_file, os.path.join(self.save_dir, file_name))
        else:
            # urlretrieve puts everything in /tmp so we strip "/tmp/" from local_file
            shutil.move(local_file, os.path.join(self.save_dir, local_file[5:]))

    def download_all_files_from_article(self, article_id):
        """Download every file in article_id

        :param article_id: ID of the Figshare article
        :type article_id: string
        """

        # Get list of dictionaries of file info
        file_list = self.list_article_files(article_id)

        # Make ./figshare/ if not available
        self.make_save_dir()

        # Save the files
        for file_dict in file_list:
            urlretrieve(
                file_dict["download_url"],
                os.path.join(self.save_dir, file_dict["name"]),
            )

    def download_random_file_from_article(self, article_id):
        """Randomly download single file in article_id

        :param article_id: ID of the Figshare article
        :type article_id: string
        """

        # Get list of dictionaries of file info
        file_list = self.list_article_files(article_id)

        # Make ./figshare/ if not available
        self.make_save_dir()

        # Save the files
        file_dict = file_list[random.randint(0, len(file_list) - 1)]
        urlretrieve(
            file_dict["download_url"],
            os.path.join(self.save_dir, file_dict["name"]),
        )
