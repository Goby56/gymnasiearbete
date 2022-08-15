from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="AI",
    version="1.0.0",
    description="Gymnasiearbete AI",
    long_description=readme,
    author="Axel Brandel, Casper Bené, Karl Rosengren",
    url="https://github.com/Goby56/gymnasiearbete-ai",
    license=license
)