from setuptools import setup, find_packages

setup(
    name="UAPS",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "mlflow",
        "joblib",
        "tqdm",
        "torch",
    ],
    entry_points={
    'console_scripts': [
        'train=UAPS.train:main',
        'infer=UAPS.inference:main',
        'clear_mlruns=UAPS.clear_mlruns:main',
        'preprocess=UAPS.data_preprocessing:main',
    ],
},
    python_requires='>=3.8',
)