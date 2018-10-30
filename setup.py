from setuptools import find_packages, setup

with open("requirements.txt") as requirements_file:
    REQUIREMENTS = requirements_file.read().splitlines()[1:]

setup(
    name="tsaplay",
    version="0.2.dev0",
    author="Sean Bugeja",
    author_email="seanbugeja23@gmail.com",
    url="https://github.com/SijanC147/Msc",
    license="LICENSE",
    description="Targeted Sentiment Analysis Playground",
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    include_package_data=True,
)
