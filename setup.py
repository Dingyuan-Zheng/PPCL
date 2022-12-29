from setuptools import setup, find_packages


setup(name='PPCL',
      version='1.0.0',
      install_requires=[
          'numpy', 'torch==1.7.0', 'torchvision==0.8.0',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss-gpu'],
      packages=find_packages(),
      keywords=[
          'Contrastive Learning',
          'Person Re-identification',
          'Plausible Proxy'
      ])

