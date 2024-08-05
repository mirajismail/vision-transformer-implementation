from setuptools import setup, find_packages

setup(
    name='vit-pytorch',
    version='0.1.0',
    description='Vision Transformer (ViT) in PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='mirajismail.2003@gmail.com',
    url='https://github.com/yourusername/vit-pytorch',
    packages=find_packages(),
    install_requires=[
        'torch>=1.6',
        'einops>=0.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
