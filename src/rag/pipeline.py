"""
src/rag/pipeline.py

RAG pipeline migrated from Colab into a proper enterprise class.
Handles ChromaDB, Hybrid Retrieval, Reranking, and HyDE.
"""

import os
import pickle
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

class RAGPipeline:
    def __init__(
        self,
        nodes_path:       str = "data/rag_data/nodes.pkl",
        config_path:      str = "data/rag_data/config.pkl",
        chroma_path:      str = "data/chroma_db",
        collection_name:  str = "pdf_rag_collection",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        reranker_model:   str = "BAAI/bge-reranker-base",
        groq_llm          = None,
    ):
        # Create directories if they don't exist
        Path("data/rag_data").mkdir(parents=True, exist_ok=True)
        
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                cfg = pickle.load(f)
            embed_model_name = cfg.get("embed_model_name", embed_model_name)
            reranker_model   = cfg.get("reranker_model",   reranker_model)
            chroma_path      = cfg.get("chroma_path",      chroma_path)
            collection_name  = cfg.get("collection_name",  collection_name)

        self.chroma_path     = chroma_path
        self.collection_name = collection_name
        self.reranker_model  = reranker_model
        self.nodes_path      = nodes_path
        self.nodes           = []

        logger.info("Loading RAG embedding model...")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, device="cpu")

        if groq_llm is not None:
            from llama_index.llms.langchain import LangChainLLM
            LlamaSettings.llm = LangChainLLM(llm=groq_llm)
        LlamaSettings.embed_model = self.embed_model

        self._chroma_client = chromadb.PersistentClient(path=chroma_path)

        if os.path.exists(nodes_path):
            with open(nodes_path, "rb") as f:
                self.nodes = pickle.load(f)
            logger.info(f"Loaded {len(self.nodes)} nodes from {nodes_path}")
            self.query_engine = self._build_engine(self.nodes)
        else:
            logger.warning("No RAG nodes found. Please upload documents to build the index.")
            self.query_engine = None

    def _build_engine(self, nodes):
        collection      = self._chroma_client.get_or_create_collection(self.collection_name)
        vector_store    = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # If building from existing vector store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

        vec_retriever  = VectorIndexRetriever(index=index, similarity_top_k=4)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)
        hybrid         = QueryFusionRetriever(
            [vec_retriever, bm25_retriever],
            similarity_top_k=4, num_queries=1, mode="reciprocal_rerank",
        )
        reranker    = FlagEmbeddingReranker(model=self.reranker_model, top_n=3)
        hyde        = HyDEQueryTransform(include_original=True)
        base_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid, node_postprocessors=[reranker], streaming=False
        )
        return TransformQueryEngine(query_engine=base_engine, query_transform=hyde)

    def query(self, text: str) -> dict:
        if not self.query_engine:
            return {"answer": "RAG engine not initialized. Upload documents first.", "sources": []}
            
        response = self.query_engine.query(text)
        answer   = response.response if hasattr(response, "response") else str(response)

        sources = []
        for i, node in enumerate(response.source_nodes):
            meta  = node.node.metadata or {}
            score = round(node.score, 3) if node.score else 0
            sources.append({
                "index":   i + 1,
                "file":    meta.get("file_name", meta.get("source", f"chunk_{i}")),
                "score":   score,
                "preview": node.node.get_text()[:200].replace("\n", " "),
            })
        return {"answer": answer, "sources": sources}

    def rebuild_from_pdfs(self, pdf_dir: str) -> int:
        """Parses new PDFs, rebuilds Chroma collection, and saves nodes."""
        from llama_parse import LlamaParse
        from llama_index.core import SimpleDirectoryReader

        logger.info(f"Rebuilding RAG index from PDFs in {pdf_dir}...")
        parser = LlamaParse(
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=False,
            num_workers=2
        )
        new_docs = SimpleDirectoryReader(pdf_dir, file_extractor={".pdf": parser}).load_data()

        node_parser = MarkdownNodeParser(chunk_size=1024, chunk_overlap=128)
        new_nodes = node_parser.get_nodes_from_documents(new_docs)

        # Clear old database
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self._chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(new_nodes, storage_context=storage_context, embed_model=self.embed_model)

        self.nodes = new_nodes
        self.query_engine = self._build_engine(new_nodes)

        with open(self.nodes_path, "wb") as f:
            pickle.dump(new_nodes, f)

        logger.success(f"Successfully rebuilt RAG with {len(new_nodes)} nodes.")
        return len(new_nodes)