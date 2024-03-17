import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rcpl",
    version="1.0.0",
    author="Martin Kovanda",
    author_email="kovanda.physics@gmail.com",
    description="Random Cyclic Plastic Loading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
