#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
