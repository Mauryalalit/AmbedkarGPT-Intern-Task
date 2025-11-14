import argparse
import os
import shutil
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM


def build_vectorstore(speech_path: Path, persist_directory: Path) -> Chroma:
    """
    Load the raw speech text, split it into overlapping chunks, embed each chunk,
    and persist it to a local Chroma directory. This is the “indexing” phase.
    """
    if not speech_path.exists():
        raise FileNotFoundError(f"Could not find the speech at {speech_path}")

    # 1) Read the speech from disk
    loader = TextLoader(str(speech_path), encoding="utf-8")
    documents = loader.load()

    # 2) Break the speech into manageable pieces.
    #    Overlap helps keep context intact when the model answers.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    # 3) Turn text into dense vectors using a small, fast sentence-transformer.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        show_progress=True,
    )

    # Re-create the DB fresh on each run so the results always match the file.
    if persist_directory.exists():
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        collection_name="ambedkar-speech",
        persist_directory=str(persist_directory),
    )
    vectorstore.persist()
    return vectorstore


def build_chain(vectorstore: Chroma) -> RetrievalQA:
    """
    Wire the retriever into a RetrievalQA chain that calls the local LLM model.
    LangChain hides the boilerplate, so we really just pass the vectorstore retriever.
    Using llama3.2:1b as it requires less memory than mistral.
    """
    llm = OllamaLLM(
    model="mistral",
    base_url="http://localhost:11434"  # forces correct engine
)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return chain


def interactive_cli(chain: RetrievalQA) -> None:
    """
    Tiny REPL loop. Ask a question, fetch a response, show the chunks used.
    Keeping it super simple so the focus stays on the RAG pipeline itself.
    """
    print("AmbedkarGPT Prototype – ask a question about the speech. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            print("Please enter a question or 'exit'.")
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        response = chain.invoke({"query": question})
        print(f"\nAnswer: {response['result']}")

        sources = response.get("source_documents", [])
        if sources:
            print("\nContext used:")
            for idx, doc in enumerate(sources, 1):
                snippet = doc.page_content.replace("\n", " ")
                print(f"  [{idx}] {snippet[:200]}{'...' if len(snippet) > 200 else ''}")


def parse_args() -> argparse.Namespace:
    """Expose a couple of flags so it’s easy to swap files or Chroma directories."""
    parser = argparse.ArgumentParser(
        description="CLI RAG prototype for Dr. B.R. Ambedkar's speech."
    )
    parser.add_argument(
        "--speech-path",
        type=Path,
        default=Path("speech.txt"),
        help="Path to the speech transcript.",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=Path("chroma_db"),
        help="Directory where Chroma will persist its data.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    # Build the index on every run (the text is tiny, so this is fast enough).
    vectorstore = build_vectorstore(args.speech_path, args.persist_directory)
    qa_chain = build_chain(vectorstore)
    interactive_cli(qa_chain)


if __name__ == "__main__":
    main()