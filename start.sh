#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
PORT=${PORT:-10000}
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
