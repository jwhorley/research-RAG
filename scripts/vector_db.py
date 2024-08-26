import asyncio
import glob
import os
from typing import List

from constants import OPENAI_EMBEDDINGS_MODEL
from constants import BAAI_EMBEDDINGS_MODEL
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import MarkdownElementNodeParser, MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from parse_json import PDFJson
from qdrant_client import QdrantClient

load_dotenv()


client = QdrantClient(
    url="{{***URL NEEDS TO GO HERE***}}",
    api_key=os.getenv("QDRANT_API_KEY"),
)
vector_store = QdrantVectorStore(
    client=client, collection_name="big-ifml-collection", batch_size=30
)


def clear_vector_store(vector_store: QdrantVectorStore):
    client: QdrantClient = vector_store.client
    client.delete_collection(collection_name=vector_store.collection_name)


def get_documents(folder_path: str) -> List[Document]:
    parsed_files = glob.glob(f"{folder_path}/*.json")
    pdf_jsons = [PDFJson.from_json_file(f) for f in parsed_files]

    return [
        Document(
            metadata={"filename": pdf_json.basename, "title": pdf_json.title},
            text=pdf_json.full_md,
        )
        for pdf_json in pdf_jsons
    ]


from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel

# Embeddings:
# Testing with OpenAI's model, AND BAAI/bdg-m3: https://huggingface.co/BAAI/bge-m3 "BGEM3FlagModel"
embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL)
#embedder = BGEM3FlagModel(BAAI_EMEDDINGS_MODEL, use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


async def main():
    node_parser = MarkdownNodeParser()
    all_nodes = node_parser.get_nodes_from_documents(get_documents("./parse-output"))

    nodes = [
        node for node in all_nodes if len(node.get_content(metadata_mode="none")) > 70
    ]

    print(f"Embedding {len(nodes)} nodes...")
    embeddings = await embedder.aembed_documents(
        [node.get_content(metadata_mode="all") for node in nodes]
    )
    print(f"Embedded {len(nodes)} nodes...")

    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    res = vector_store.add(nodes)
    # print(res)


if __name__ == "__main__":
    asyncio.run(main())
