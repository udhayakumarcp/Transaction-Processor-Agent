# Project Setup Guide

## Create the Virtual Environment

To create a virtual environment, run the following command:

```bash
python3 -m venv .venv
```

## Activate the Virtual Environment

Run the following command based on your OS:

```bash
source .venv/bin/activate
```

## Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Update the Dependencies List

Whenever you install, uninstall, or upgrade packages, update requirements.txt by running:

```bash
pip freeze > requirements.txt
```

## Run the application

To run the application, execute:

```bash
streamlit run main.py

```
