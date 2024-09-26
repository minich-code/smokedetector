
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


__version__ = "0.0.1"
SRC_REPO = "smokedetector"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=SRC_REPO,
    version=__version__,
    license='MIT',
    url='https://github.com/minich-code/smokedetector.git',
    download_url='https://github.com/minich-code/smokedetector.git',
    description='This is project to detect smokes in your home',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Western',
    author_email='minichworks@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    keywords='deep learning smart home smoke detector IoT',
    entry_points={
        'console_scripts': [
            'smokedetector=smokedetector.scripts.smokedetector:main',
        ],
    },
)


