# FigshareDownloader

The FigshareDownloader class provides functions to query and download files associated with a [Figshare](https://figshare.com/) article.

**Example Usage:**

```python
articleID = 12345678

fsdl = FigshareDownloader()

# This is a list of dictionaries of files and information about them
files = fsdl.listFiles(articleID)

# Download the last file from the article to the figshare/ directory
fsdl.downloadSingleFile(articleID, files[0])

# Download every file from the article to the figshare/ directory
fsdl.downloadAllFiles(articleID)
```

| Parameter   | Description                                 |
|-------------|---------------------------------------------|
| `directory` | An optional directory to download files to. |
