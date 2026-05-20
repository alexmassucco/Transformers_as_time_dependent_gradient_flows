from setuptools import setup, find_packages

setup(
    name="attn-sphere",
    version="0.1.0",
    description="Official experiments for Multi-Headed Transformer Architectures as Time-dependent Wasserstein Gradient Flows",
    packages=find_packages(exclude=("experiments",)),
    install_requires=[],
)
