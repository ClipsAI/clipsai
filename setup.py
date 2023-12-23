from setuptools import setup

setup(
    name='clipsai',
    version='0.1',
    description='A Python package for clipping video content',
    url='https://github.com/ClipsAI/clip',
    author='ClipsAI',
    author_email='support@clipsai.com',
    license='MIT',
    packages=['clipsai'],
    # we need to go through pyproject.toml and determine what dependencies we need
    install_requires=[
        'numpy',
        'opencv-python',
        'mediapipe',
        'scipy',
        'torch',
        'nltk',
        'python-magic',
        'reportlab',
        'whisperx@git+https://github.com/m-bain/whisperx.git',
        'sentence-transformers',
        'matplotlib',
        'av',
        'cv2',
        'pynvml',
        'scenedetect'
        'facenet-pytorch',
        'pyannote.audio',
        'pyannote.core',
    ],
    zip_safe=False
)
