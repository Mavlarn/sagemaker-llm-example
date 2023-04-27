import os
import pinecone 
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain.text_splitter import SpacyTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

#一些配置文件
openai_key="你的key" # 注册 openai.com 后获得

#科学上网你懂得
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

#初始化pinecone
pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_environment
)
index = pinecone.Index(pinecone_index)

#初始化OpenAI的embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

#初始化text_splitter
text_splitter = SpacyTextSplitter(pipeline='zh_core_web_sm',chunk_size=500,chunk_overlap=50)

# 读取目录下所有后缀是txt的文件
loader = DirectoryLoader('../docs', glob="**/*.txt", loader_cls=TextLoader)

#读取文本文件
documents = loader.load()

# 使用text_splitter对文档进行分割
split_text = text_splitter.split_documents(documents)
try:
	for document in tqdm(split_text):
		# 获取向量并储存到pinecone
		Pinecone.from_documents([document], embeddings, index_name=pinecone_index)
except Exception as e:
    print(f"Error: {e}")
    quit()


