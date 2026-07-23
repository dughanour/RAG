# RAG

uvicorn main:app --reload --host 0.0.0.0 --port 8000
streamlit run view/view.py --server.port 8501
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc