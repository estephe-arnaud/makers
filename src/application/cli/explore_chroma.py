#!/usr/bin/env python3
"""
Simple script to explore ChromaDB - similar to mongosh but for ChromaDB
Usage: python -m src.application.cli.explore_chroma <path_to_chromadb> [--collection COLLECTION_NAME]
"""
import argparse
import chromadb
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Explore a ChromaDB database"
    )
    parser.add_argument(
        "chroma_path",
        type=str,
        help="Path to ChromaDB directory (e.g., data/chroma_db)"
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=None,
        help="Collection name to explore (default: first collection found)"
    )
    
    args = parser.parse_args()
    
    chroma_path = Path(args.chroma_path).resolve()
    
    if not chroma_path.exists():
        print(f"❌ Error: Path '{chroma_path}' does not exist")
        return
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    print("=" * 80)
    print("📦 ChromaDB DATABASE")
    print("=" * 80)
    print(f"Path: {chroma_path}\n")
    
    # List all collections
    print("📚 Available collections:")
    collections = client.list_collections()
    if not collections:
        print("  No collections found")
        return
    
    for coll in collections:
        print(f"  - {coll.name} ({coll.count()} documents)")
    
    # Select collection
    collection_name = args.collection
    if collection_name is None:
        if collections:
            collection_name = collections[0].name
            print(f"\n⚠️  No collection specified, using: {collection_name}")
        else:
            print("\n❌ No collections available")
            return
    else:
        # Verify that the collection exists
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            print(f"\n❌ Error: Collection '{collection_name}' does not exist")
            print(f"Available collections: {', '.join(collection_names)}")
            return
    
    print(f"\n🔍 Active collection: {collection_name}")
    collection = client.get_collection(name=collection_name)
    print(f"📊 Document count: {collection.count()}\n")
    
    # Display some documents
    print("=" * 80)
    print("📄 First documents:")
    print("=" * 80)
    
    results = collection.get(limit=5)
    
    for i, (doc_id, doc_text, metadata) in enumerate(zip(
        results['ids'],
        results.get('documents', [''] * len(results['ids'])),
        results.get('metadatas', [{}] * len(results['ids']))
    ), 1):
        print(f"\n--- Document {i} ---")
        print(f"ID: {doc_id}")
        print(f"Arxiv ID: {metadata.get('arxiv_id', 'N/A')}")
        print(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}")
        print(f"Text (first 200 characters):")
        print(f"  {doc_text[:200]}...")
        print(f"Full metadata: {metadata}")
    
    # List unique arxiv_ids
    print("\n" + "=" * 80)
    print("📚 Unique Arxiv IDs:")
    print("=" * 80)
    
    all_results = collection.get()
    arxiv_ids = set()
    for metadata in all_results.get('metadatas', []):
        if 'arxiv_id' in metadata:
            arxiv_ids.add(metadata['arxiv_id'])
    
    for arxiv_id in sorted(arxiv_ids):
        count = sum(1 for m in all_results.get('metadatas', []) 
                    if m.get('arxiv_id') == arxiv_id)
        print(f"  {arxiv_id}: {count} chunks")
    
    # Interactive function to search
    print("\n" + "=" * 80)
    print("💡 To search for a specific document:")
    print("=" * 80)
    print("""
# Example: search by arxiv_id
results = collection.get(
    where={"arxiv_id": "2410.23831"},
    limit=5
)

# Example: search by chunk_id
results = collection.get(
    where={"chunk_id": "2410.23831_chunk_001"},
    limit=1
)

# Example: vector search (similarity)
results = collection.query(
    query_texts=["What are the latest advancements in face analysis"],
    n_results=3
)
""")


if __name__ == "__main__":
    main()

