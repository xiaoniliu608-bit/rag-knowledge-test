import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import dashscope
# 新增解析库
import PyPDF2
from docx import Document

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
app = FastAPI(title="公司知识库RAG系统")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ========== 全局变量 ==========
vector_db = None
rag_ready = False
# 本地Embedding（不需要阿里云密钥，直接用）
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
UPLOAD_DIR = "uploads"
DB_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ========== 【新增：解析函数】 ==========
def parse_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_docx(file_path: str) -> str:
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()



# ========== 初始化向量库 ==========
def init_rag():
    global vector_db
    try:
        # 尝试加载已保存的向量库
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            vector_db = FAISS.load_local(
                DB_DIR, embeddings, allow_dangerous_deserialization=True
            )
            print("✅ 成功加载已保存的向量库")
        else:
            # 第一次运行，创建空向量库
            vector_db = FAISS.from_texts(["初始化知识库"], embeddings)
            vector_db.save_local(DB_DIR)
            print("✅ 已创建新的向量库")
    except Exception as e:
        print(f"⚠️ 向量库初始化失败: {e}")
        vector_db = None
# 启动时执行初始化
init_rag()
# ========== 健康检查接口 ==========
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": rag_ready,
        "vector_db_loaded": vector_db is not None
    }
# ========== 【修改后的上传接口】 ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db, rag_ready
    try:
        # 保存上传的文件
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        
        # 根据后缀选择解析器
        filename_low = file.filename.lower()
        if filename_low.endswith(".txt"):
            full_text = parse_txt(save_path)
        elif filename_low.endswith(".pdf"):
            full_text = parse_pdf(save_path)
        elif filename_low.endswith(".docx"):
            full_text = parse_docx(save_path)
        else:
            return {"error": "目前只支持txt、pdf、docx文件"}

        if not full_text.strip():
            return {"error": "文件内容为空或者解析不出文字"}
        # 文本切分
        chunks = text_splitter.split_text(full_text)
        print(f"📄 切分出 {len(chunks)} 个文本块")
        # 构建/追加向量库
        if vector_db is None:
            vector_db = FAISS.from_texts(chunks, embeddings)
        else:
            vector_db.add_texts(chunks)
        # 保存到磁盘
        vector_db.save_local(DB_DIR)
        rag_ready = True
        return {
            "msg": "✅ 上传成功，知识库已更新",
            "filename": file.filename,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}
# ========== 问答接口 ==========
class ChatRequest(BaseModel):
    query: str
@app.post("/chat")
async def chat(req: ChatRequest):
    global vector_db
    try:
        if vector_db is None:
            return {"answer": "❌ 知识库为空，请先上传文档！"}
        # 检索相关片段
        docs = vector_db.similarity_search(req.query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        # 优化Prompt，强制提炼总结
        prompt = """
你是企业知识库助手，请严格依据【参考文档】内容回答用户问题。
要求：
1. 只能使用文档内信息，禁止编造不存在的内容；
2. 精简提炼答案，不要直接复制大段原文；
3. 不要输出检索标记、序号、多余联系方式；
4. 如果文档没有相关信息，直接回复：知识库暂时没有该问题相关资料。
【参考文档】
{context}
用户问题：{query}
""".format(context=context, query=req.query)
         # 调用阿里云通义千问 qwen-turbo
        resp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[{"role":"user", "content": prompt}]
        )
        print(f"dashscope完整返回：{resp}")
        if resp.code == 200:
            answer = resp.output.text
        else:
            answer = f"接口调用失败，code={resp.code}, msg={resp.message}"
        return {
            "query": req.query,
            "answer": answer,
            "retrieved_docs": len(docs)
        }
    except Exception as e:
        print(f"LLM调用异常：{str(e)}")
        return {"error": f"问答失败: {str(e)}"}
# ========== 评估接口（先放着，后面再调） ==========
class EvaluateRequest(BaseModel):
    samples: list
@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    return {"msg": "评估接口已就绪，样本数: " + str(len(req.samples))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
