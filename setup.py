from setuptools import find_packages, setup

setup(
    name="clipsai",
    py_modules=["clipsai"],
    version="0.2.1",
    description=(
        "Clips AI is an open-source Python library that automatically converts long "
        "videos into clips"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Smidt, Johann Ramirez, Armel Talla",
    author_email="support@clipsai.com",
    url="https://clipsai.com/",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "av",
        "facenet-pytorch",
        "matplotlib",
        "mediapipe",
        "nltk",
        "numpy",
        "opencv-python",
        "pandas",
        "psutil",
        "pyannote.audio",
        "pyannote.core",
        "pynvml",
        "pytest",
        "python-magic",
        "scenedetect",
        "scikit-learn",
        "sentence-transformers",
        "scipy",
        "torch",
    ],
    zip_safe=False,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://docs.clipsai.com/",
        "Homepage": "https://clipsai.com/",
        "Repository": "https://github.com/ClipsAI/clipsai",
        "Issues": "https://github.com/ClipsAI/clipsai/issues",
    },
    include_package_data=True,
    extras_require={
        "dev": [
            "black",
            "black[jupyter]",
            "build",
            "flake8",
            "ipykernel",
            "pytest",
            "twine",
        ],
    },
)
