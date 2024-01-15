import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peaknet",
    version="24.01.15",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Masked Autoencoder for X-ray Image Encoding (MAXIE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/peaknet",
    keywords = ['Foundation model', 'Masked-Autoencoder', 'X-ray Diffraction Image Encoder'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
