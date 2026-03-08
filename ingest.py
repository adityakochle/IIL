"""One-shot ingestion script.

Loads all documents, creates parent-child chunks, embeds child nodes,
and upserts into Qdrant.

Usage:
  python ingest.py
  python ingest.py --reset   # Drops and recreates the Qdrant collection
"""
import argparse
import sys
import time


def main(reset: bool = False) -> None:
    print("\n" + "=" * 60)
    print("  IIL DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    from config import settings

    # Validate API keys
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    if "localhost" not in settings.qdrant_url and not settings.qdrant_api_key:
        print("WARNING: QDRANT_API_KEY not set. Connecting to Qdrant without auth.")

    t_start = time.time()

    # Step 1: Load documents
    print("\n[Step 1/3] Loading documents...")
    from pipelines.loaders import load_all_documents
    documents = load_all_documents()

    # Step 2: Chunk into parent-child pairs
    print("[Step 2/3] Creating parent-child chunks...")
    from pipelines.chunkers import create_parent_child_nodes
    child_nodes, parent_store = create_parent_child_nodes(documents)

    # Step 3: Embed and index
    print("[Step 3/3] Embedding and indexing to Qdrant...")
    from pipelines.indexer import _get_qdrant_client, ensure_collection, index_nodes

    if reset:
        print(f"  Dropping collection '{settings.qdrant_collection}'...")
        client = _get_qdrant_client()
        try:
            client.delete_collection(settings.qdrant_collection)
            print("  Collection dropped.")
        except Exception as e:
            print(f"  Note: {e}")

    index = index_nodes(child_nodes)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Documents loaded:    {len(documents)}")
    print(f"  Parent chunks:       {len(parent_store)}")
    print(f"  Child nodes indexed: {len(child_nodes)}")
    print(f"  Collection:          {settings.qdrant_collection}")
    print(f"  Time elapsed:        {elapsed:.1f}s")
    print("=" * 60)
    print("\nNext steps:")
    print("  python main.py --interactive   # Start the IIL query interface")
    print("  python main.py --query '...'   # Single query mode")
    print("  python evals/benchmark.py      # Run accuracy benchmark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IIL document ingestion pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()
    main(reset=args.reset)
