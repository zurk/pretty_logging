"""Script to setup pretty_logging package."""

import os

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pretty_logging",
    description="Package to make logging prettier.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.2.0",
    url="https://github.com/zurk/pretty_logging",
    download_url="https://github.com/zurk/pretty_logging",
    packages=["pretty_logging"],
    install_requires=[
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
)
