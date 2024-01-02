from setuptools import setup, find_packages

setup(
    name='clipsai',
    version='0.1',
    description='Clips AI is an open-source Python library that automatically converts long videos into clips',
    url='https://docs.clipsai.com/',
    author='Clips AI',
    author_email='support@clipsai.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'av',
        'facenet-pytorch',
        'matplotlib',
        'mediapipe',
        'nltk',
        'numpy',
        'opencv-python',
        'pandas'
        'psutil',
        'pyannote.audio',
        'pyannote.core',
        'pynvml',
        'pytest',
        'python-magic',
        'scenedetect',
        'scikit-learn',
        'sentence-transformers',
        'scipy',
        'torch', 
    ],
    zip_safe=False
)
