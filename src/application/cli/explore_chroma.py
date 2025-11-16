#!/usr/bin/env python3
"""
Script simple pour explorer ChromaDB - similaire à mongosh mais pour ChromaDB
Usage: python -m src.application.cli.explore_chroma <path_to_chromadb> [--collection COLLECTION_NAME]
"""
import argparse
import chromadb
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Explorer une base de données ChromaDB"
    )
    parser.add_argument(
        "chroma_path",
        type=str,
        help="Chemin vers le dossier ChromaDB (ex: data/chroma_db)"
    )
    parser.add_argument(
        "-c", "--collection",
        type=str,
        default=None,
        help="Nom de la collection à explorer (par défaut: première collection trouvée)"
    )
    
    args = parser.parse_args()
    
    chroma_path = Path(args.chroma_path).resolve()
    
    if not chroma_path.exists():
        print(f"❌ Erreur: Le chemin '{chroma_path}' n'existe pas")
        return
    
    # Connexion à ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    print("=" * 80)
    print("📦 BASE DE DONNÉES ChromaDB")
    print("=" * 80)
    print(f"Chemin: {chroma_path}\n")
    
    # Lister toutes les collections
    print("📚 Collections disponibles:")
    collections = client.list_collections()
    if not collections:
        print("  Aucune collection trouvée")
        return
    
    for coll in collections:
        print(f"  - {coll.name} ({coll.count()} documents)")
    
    # Sélectionner la collection
    collection_name = args.collection
    if collection_name is None:
        if collections:
            collection_name = collections[0].name
            print(f"\n⚠️  Aucune collection spécifiée, utilisation de: {collection_name}")
        else:
            print("\n❌ Aucune collection disponible")
            return
    else:
        # Vérifier que la collection existe
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            print(f"\n❌ Erreur: La collection '{collection_name}' n'existe pas")
            print(f"Collections disponibles: {', '.join(collection_names)}")
            return
    
    print(f"\n🔍 Collection active: {collection_name}")
    collection = client.get_collection(name=collection_name)
    print(f"📊 Nombre de documents: {collection.count()}\n")
    
    # Afficher quelques documents
    print("=" * 80)
    print("📄 Premiers documents:")
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
        print(f"Texte (premiers 200 caractères):")
        print(f"  {doc_text[:200]}...")
        print(f"Métadonnées complètes: {metadata}")
    
    # Lister les arxiv_id uniques
    print("\n" + "=" * 80)
    print("📚 Arxiv IDs uniques:")
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
    
    # Fonction interactive pour chercher
    print("\n" + "=" * 80)
    print("💡 Pour chercher un document spécifique:")
    print("=" * 80)
    print("""
# Exemple: chercher par arxiv_id
results = collection.get(
    where={"arxiv_id": "2410.23831"},
    limit=5
)

# Exemple: chercher par chunk_id
results = collection.get(
    where={"chunk_id": "2410.23831_chunk_001"},
    limit=1
)

# Exemple: recherche vectorielle (similarité)
results = collection.query(
    query_texts=["What are the latest advancements in face analysis"],
    n_results=3
)
""")


if __name__ == "__main__":
    main()

