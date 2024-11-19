#!/bin/bash
# Anaconda 가상 환경 활성화 (Azure에서는 /home/site/conda/envs/ 경로로 되어 있을 가능성 있음)
source /opt/conda/bin/activate /home/site/conda/envs/yoloCuda_py311  # Anaconda 환경 활성화

# Streamlit 실행 (80번 포트에서 실행)
pip install streamlit
streamlit run app.py --server.address=0.0.0.0 --server.port=80