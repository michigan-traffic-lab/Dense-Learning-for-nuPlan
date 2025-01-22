from setuptools import setup, find_packages

setup(
    name="safedriver-nuplan",
    version="0.1.0",
    description="A short description of your package",
    author="Haojie Zhu",
    author_email="zhuhj@umich.edu",
    url="https://github.com/yourusername/your-repo",
    packages=["safedriver_nuplan"],
    install_requires=[
        # List your package dependencies here
        # e.g., 'numpy', 'requests'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
