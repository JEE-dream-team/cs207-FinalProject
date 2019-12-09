# setup.py


from setuptools import setup, find_packages

setup(
    name="jeeautodiff",
    url="https://github.com/JEE-dream-team/cs207-FinalProject.git",
    author="JEE",
    packages=find_packages(),
    install_requires=["numpy"],
    version="0.1",
    license="MIT",
    description="Automatic differentiation package",
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],    
    python_requires='>=3.6'
)
