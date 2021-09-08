import json
import os
from urllib.request import urlretrieve

import requests
from requests.exceptions import HTTPError


class FigshareDownloader:
    """Makes HTTP requests for images on Figshare"""

    def __init__(self, directory="figshare/"):
        self.saveDir = os.path.join(os.getcwd(), directory)

    def makeUrl(self, articleID):
        """Form the full URL for requests"""
        return "https://api.figshare.com/v2/" + f"articles/{articleID}/files"

    def makeSaveDir(self):
        # Don't throw errors if we download stuff multiple times
        os.makedirs(self.saveDir, exist_ok=True)

    def sendRequest(self, url, headers):
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

    def listFiles(self, articleID):
        """Get a dictionary of files from articleID.

        :param articleID: ID of Figshare article
        :type articleID: str
        """

        requestUrl = self.makeUrl(articleID)
        requestHeader = {"Content-Type": "application/json"}
        response = self.sendRequest(requestUrl, headers=requestHeader)
        return response

    def downloadFile(self, articleID, fileDict):
        """Helper function to download files from articleID, save to ./figshare/

        :param articleID: ID of the Figshare article
        :type articleID: string
        :param fileDict: dictionary of Figshare file info
        :type fileDict: dictionary
        """

        urlretrieve(
            fileDict["download_url"],
            os.path.join(self.saveDir, fileDict["name"]),
        )

    def downloadSingleFile(self, articleID, fileDict):
        """Download one file in articleID"""

        # Make ./figshare/ if not available
        self.makeSaveDir()

        # Save the file
        self.downloadFile(articleID, fileDict)

    def downloadAllFiles(self, articleID):
        """Download every file in articleID"""

        # Get list of dictionaries of file info
        fileList = self.listFiles(articleID)

        # Make ./figshare/ if not available
        self.makeSaveDir()

        # Save the files
        for fileDict in fileList:
            self.downloadFile(articleID, fileDict)
