from setuptools import setup, find_packages

setup(
    name="attn-sphere",
    version="0.1.0",
    description="Multi-head attention gradient flows on the sphere",
    packages=find_packages(exclude=("experiments",)),
    install_requires=[],
)
