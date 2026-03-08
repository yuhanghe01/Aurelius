from setuptools import setup, find_packages

setup(
    name="aurelius",
    version="0.1.0",
    description="Relation Aware Text-to-Audio Generation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "einops>=0.6.0",
    ],
)
