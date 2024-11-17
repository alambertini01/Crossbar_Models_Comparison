from setuptools import setup, find_packages
from setuptools.command.install import install
import os

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Define custom post-installation command
class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # Run the standard install command
        super().run()
        # Execute custom commands after installation
        os.system("pyspice-post-installation --install-ngspice-dll")
        os.system("pyspice-post-installation --check-install")
        

# Setup function
setup(
    name="Crossbar_Models_Comparison", 
    version="0.1.0",  # Package version
    author="Alessandro Lambertini",
    author_email="alessandro.lambertini6@gmail.com",
    description="benchmarks and compares various models of parasitic resistances found in the literature",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alambertini01/Crossbar_Models_Comparison",
    packages=find_packages(),  # Automatically find all packages
    install_requires=requirements,  # Install dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
    cmdclass={
        'install': PostInstallCommand,  # Link custom post-install command
    },
)
