from setuptools import setup, find_packages

setup(
    name="pirl_carla",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "carla",
        "optuna",
        "pygame"
    ],
    author="Shashank Raj",
    description="Physics-Informed Reinforcement Learning for Autonomous Driving",
    python_requires=">=3.7",
)
