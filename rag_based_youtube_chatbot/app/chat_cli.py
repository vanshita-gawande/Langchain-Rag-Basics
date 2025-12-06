# app/chat_cli.py
from .ingest import index_youtube_video
from .rag_chain import build_rag_chain


def main():
    print("=== YouTube RAG Chatbot ===")
    video_url = input("Enter YouTube URL: ").strip()

    # Use video ID as collection name, but you can simplify:
    from langchain_community.document_loaders.youtube import YoutubeLoader

    video_id = YoutubeLoader.extract_video_id(video_url)
    collection_name = f"yt_{video_id}"

    print(f"\n[1/2] Indexing video (this is done once per video)...")
    index_youtube_video(video_url, collection_name)

    print("[2/2] Building RAG chain...")
    chain = build_rag_chain(collection_name)

    print("\nYou can now ask questions about the video.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = chain.invoke( question)
        answer = result.content
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
