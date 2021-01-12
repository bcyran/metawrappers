from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
DESCRIPTION_FILE = PROJECT_ROOT / "README.md"

with REQUIREMENTS_FILE.open() as f:
    requirements = f.read().splitlines()

with DESCRIPTION_FILE.open() as f:
    long_description = f.read()

setup(
    name="metawrappers",
    version="0.1.0",
    description="Metaheuristic-based feature selection wrappers",
    author="Bazyli Cyran",
    author_email="kontakt@bazylicyran.pl",
    url="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    platforms="any",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
