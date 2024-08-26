import asyncio
from dataclasses import dataclass
import glob
import os
from parse_json import PDFJson
from pydantic import BaseModel, Field
from typing import List
from datetime import date, datetime, time, timedelta
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv
import arxiv
import aiohttp

load_dotenv()


class Author(BaseModel):
    first_name: str
    last_name: str
    affiliations: List[str]


class PaperMetadata(BaseModel):
    """Data Model for the paper metadata."""

    id: str | None = None
    title: str | None = None
    authors: List[Author]
    categories: List[str] = Field(..., min_items=6, max_items=6)
    publication_date: date | None = None
    abstract: str | None = None


llm = OpenAI(model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    """
    You are an expert at extracting metadata from research papers.
    Given the first page of a research paper, your task is to accurately extract the following metadata from the publication:

    1. Title: extract the full title of the paper
    2. Authors: extract the full name of each author and their affiliations
    - DO NOT extract the email addresses
    4. Extract any information related to affliated conferences where the paper was presented
    5. Categories / Keywords:
     - Based on the content of the first page and the Abstract, provide exactly 6 cateogires or keywords that best represent the main topics, methods, or applications discussed in the paper.
     - If possible, extract relevant phrases direclty from the abstract ot use as categories / keywords.
     - If the Abstract does not provide sufficient information, come up with your own concise and descriptive categories/keywords.
     - Choose categories/keywords that potential readers might use when searching for this paper on academic search engines like Google Scholar.

    Here is the first page of the paper:
    {text}
    """
)


@dataclass
class ArxivMeta:
    title: str
    abstract: str
    publication_date: date


async def get_arxiv_metadata(arvix_id: str) -> ArxivMeta:
    import feedparser
    import arxiv

    async with aiohttp.ClientSession() as session:
        url = f"http://export.arxiv.org/api/query?id_list={arvix_id}"
        async with session.get(url) as resp:
            if resp.status == 200:
                content = await resp.text()
                feed: feedparser.FeedParserDict = feedparser.parse(content)
                entry = feed.entries[0]
                result = arxiv.Result._from_feed_entry(entry)
                return ArxivMeta(
                    title=result.title,
                    abstract=result.summary,
                    publication_date=result.published.date(),
                )

            else:
                raise Exception(f"Failed to fetch arxiv metadata for {arvix_id}")


async def process_pdf(pdf_json: PDFJson) -> PaperMetadata:
    print(f"Processing PDF {pdf_json.basename}")
    llm_task = asyncio.create_task(
        llm.astructured_predict(
            output_cls=PaperMetadata, prompt=prompt, text=pdf_json.pages[0].md
        )
    )
    arxiv_task = asyncio.create_task(get_arxiv_metadata(pdf_json.arxiv_id))
    llm_response: PaperMetadata = await llm_task
    try:
        arxiv_response = await arxiv_task
        final_meta = PaperMetadata(
            id=pdf_json.arxiv_id,
            title=arxiv_response.title,
            abstract=arxiv_response.abstract,
            publication_date=arxiv_response.publication_date,
            authors=llm_response.authors,
            categories=llm_response.categories,
        )
    except Exception as e:
        print(f"Failed to fetch arxiv metadata: {e}")
        final_meta = llm_response
        final_meta.id = pdf_json.arxiv_id
    return final_meta


def get_unextracted_files(input_dir: str, extractions: list[PaperMetadata]):
    input_files = glob.glob(f"{input_dir}/*.json")
    return [
        file
        for file in input_files
        if os.path.basename(file).replace(".json", "")
        not in [e.id for e in extractions]
    ]


def read_extractions(file: str) -> list[PaperMetadata]:
    with open(file, "r") as f:
        return [PaperMetadata.model_validate_json(line) for line in f.readlines()]


async def main():
    output_file = "./meta-output/meta-extractions.jsonl"
    all_output_metas = read_extractions(output_file)

    unextracted_files = get_unextracted_files("./parse-output", all_output_metas)
    pdf_jsons = [PDFJson.from_json_file(file) for file in unextracted_files]
    tasks = [process_pdf(pdf_json) for pdf_json in pdf_jsons]

    results: list[PaperMetadata] = await asyncio.gather(*tasks)
    with open(output_file, "a") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


if __name__ == "__main__":
    asyncio.run(main())