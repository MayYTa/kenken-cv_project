#!/bin/bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 \
     --server.headless true --browser.gatherUsageStats false