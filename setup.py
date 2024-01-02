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
        'numpy',
        'opencv-python',
        'mediapipe',
        'scipy',
        'torch',
        'nltk',
        'python-magic',
        'sentence-transformers',
        'matplotlib',
        'av',
        'opencv-python',
        'pynvml',
        'scenedetect',
        'facenet-pytorch',
        'pyannote.audio',
        'pyannote.core',
        'onnxruntime',
        'pytest',
        'pytest-mock',
        'psutil',
        'pandas'
        'onnxruntime',
        'scikit-learn',
    ],
    zip_safe=False
)
