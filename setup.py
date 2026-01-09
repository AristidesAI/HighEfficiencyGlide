"""
HighEfficiencyGlide - Solar-powered autonomous glider simulation and control.

This package provides tools for:
- Aerodynamic simulation and wing optimization
- Solar cell placement and energy balance modeling
- ArduPilot integration for autonomous flight
- YOLO-based vision systems for GPS-denied navigation
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="high-efficiency-glide",
    version="0.1.0",
    description="Solar-powered autonomous glider simulation and control system",
    author="HighEfficiencyGlide Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
        ],
        "jetson": [
            "tensorrt",
        ],
    },
    entry_points={
        "console_scripts": [
            "heg-simulate=simulation.main:main",
            "heg-optimize=simulation.optimization.genetic_algorithm:main",
            "heg-ground-station=flight_control.ground_station.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
