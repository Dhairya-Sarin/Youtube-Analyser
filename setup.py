from setuptools import setup, find_packages

setup(
    name="youtube-analytics-pro",
    version="2.0.0",
    description="Advanced AI-powered YouTube analytics with comprehensive feature extraction and predictive modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="YouTube Analytics Pro Team",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "shap>=0.42.1",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.15.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.0",
        "opencv-python>=4.8.0.76",
        "Pillow>=10.0.0",
        "textblob>=0.17.1",
        "nltk>=3.8.1",
        "openpyxl>=3.1.2",
        "beautifulsoup4>=4.12.2",
        "python-dateutil>=2.8.2",
        "joblib>=1.3.2",
    ],
    extras_require={
        "all": [
            "librosa>=0.10.0",
            "SpeechRecognition>=3.10.0",
            "yt-dlp>=2023.7.6",
            "transformers>=4.30.0",
            "spacy>=3.6.0",
        ],
        "development": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "youtube-analytics-pro=main:main",
        ],
    },
)