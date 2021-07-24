import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="augraphy",
    version="3.0.0",
    author="Sparkfish LLC",
    author_email="packages@sparkfish.com",
    description="Augmentation pipeline for rendering synthetic paper printing and scanning processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sparkfish/augraphy",
    project_urls={
        "Bug Tracker": "https://github.com/sparkfish/augraphy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy >= 1.20.1",
        "opencv-python >= 4.5.1.48",
        "sklearn >= 0.0",
        "scipy >= 1.6.3",
    ],
)
