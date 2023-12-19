from setuptools import setup

setup(name='clip',
      version='0.1',
      description='A Python package for clipping video content',
      url='https://github.com/ClipsAI/clip',
      author='ClipsAI',
      author_email='support@clipsai.com',
      license='GPLv3',
      packages=['clip'],
      # we need to go through pyproject.toml and determine what dependencies we need
      install_requires=[
          'numpy',
          'opencv-python',
          'mediapipe',
          'scipy',
          'torch',
          'nltk',
          'magic',
          'reportlab',
          'whisperx',
        ],
      zip_safe=False)