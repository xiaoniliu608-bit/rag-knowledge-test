# 公司知识库RAG系统
基于 FastAPI + LangChain + FAISS 的轻量本地RAG知识库项目
使用本地 all-MiniLM-L6-v2 Embedding，无需大模型API密钥即可完成文档检索。

## ✨ 功能
1. /health 健康检查接口
2. /upload 上传TXT文档，自动切片并存入FAISS向量库
3. /chat 提问，从知识库检索相关原文

## 📦 环境安装
```bash
pip install fastapi uvicorn langchain langchain-community faiss-cpu python-dotenv sentence-transformers
