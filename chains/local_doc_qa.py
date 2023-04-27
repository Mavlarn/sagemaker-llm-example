from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate, LLMChain, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory


# from models.sagemaker_llm import SagemakerLLM
from typing import Dict
import os, json
from configs.model_config import *
import datetime
from typing import List
from textsplitter import ChineseTextSplitter, CSVTextSplitter


# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3


class ContentHandler(ContentHandlerBase):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input = {"ask": prompt, **model_kwargs}
        input_str = json.dumps(input)
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["answer"]

content_handler = ContentHandler()

def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = UnstructuredFileLoader(filepath)
        csvsplitter = CSVTextSplitter()
        docs = loader.load_and_split(csvsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def get_docs_with_score(docs_with_score):
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


class LocalDocQA:
    llm: object = None
    retrievalqa_chain: object = None
    simple_chain: object = None
    llm_chain: object = None
    embeddings: object = None
    vector_store: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    medqa_vs_path: str = ""

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                #  embedding_device=EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 llm_model: str = LLM_MODEL,
                 top_k=VECTOR_SEARCH_TOP_K,
                 use_ptuning_v2: bool = USE_PTUNING_V2
                 ):
        # self.llm = SagemakerLLM()
        model_ep_name=llm_model_dict[llm_model]
        self.llm = SagemakerEndpoint(
            endpoint_name=model_ep_name,
        #         credentials_profile_name="credentials-profile-name", 
            region_name="us-east-1", 
            model_kwargs={"temperature": 0.1},
            content_handler=content_handler
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                        model_kwargs={'device': 'cuda'})


        # self.llm.history_len = llm_history_len
        PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        self.retrievalqa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            memory=ConversationBufferMemory(),
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs
        )
        self.simple_chain = load_qa_chain(
            llm=self.llm,
            prompt=PROMPT,
            memory=ConversationBufferMemory()
        )
        self.llm_chain = LLMChain(
            llm=self.llm, 
            prompt=PromptTemplate.from_template("{query}"),
            memory=ConversationBufferMemory(),
            output_key = 'result'
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': 'cuda'})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    print(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            self.vector_store = vector_store
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def get_direct_answer(self,
                         query,
                         chat_history=[]):
        response = self.llm_chain(query)
        return response
        

    def get_knowledge_based_answer(self,
                                   query,
                                   chat_history=[]):
        related_docs_with_score = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        prompt = generate_prompt(related_docs, query)

        # using retrievalqa_chain
        # response = self.retrievalqa_chain({"query": query})
        # result 包含 query, result
        # response = self.simple_chain({"query": query})
        # response.source_documents =related_docs

        # using retrievalqa_chain
        doc_content = "\n".join([doc.page_content for doc in related_docs])
        docs = [ Document(page_content=doc_content) ]
        result = self.simple_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        response = {"query": query, "result": result.output_text, "source_documents": related_docs}


        history=chat_history
        history[-1][0] = query
        return response, history

