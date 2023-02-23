from setuptools import setup

# Requirements for the Dataflow dataset creation pipeline.
setup(
    name="ppai-fire-risk",
    url="https://github.com/abner86/ai_wildfire_risk_prediction.git",
    packages=["serving"],
    install_requires=[
        "apache-beam[gcp]==2.45.0",
        "earthengine-api==0.1.342",
        "tensorflow==2.11.0",
    ],
)