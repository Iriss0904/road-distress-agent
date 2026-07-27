#!/usr/bin/env python3
"""Build Qdrant index from rag_chunks.jsonl using BGE-M3 embeddings.

Reads embed_text from each chunk, encodes with BGE-M3 (dense + sparse),
and upserts into Qdrant. Idempotent: re-running overwrites existing points.

Usage:
    python scripts/build_qdrant_index.py <doc_dir> [--batch-size N] [--recreate]

Example:
    python scripts/build_qdrant_index.py data/processed/my_document
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

_DENSE_DIM = 1024


def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _ensure_collection(client: QdrantClient, collection: str, recreate: bool) -> None:
    if recreate and client.collection_exists(collection_name=collection):
        client.delete_collection(collection_name=collection)
        print(f"Deleted collection: {collection}")
    if not client.collection_exists(collection_name=collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=_DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )
        print(f"Created collection: {collection}")
    else:
        print(f"Collection exists: {collection} — will upsert (overwrite existing points)")

    # Ensure keyword index on semantic_role for efficient filter-based retrieval.
    # create_payload_index is idempotent — safe to call on every run.
    client.create_payload_index(
        collection_name=collection,
        field_name="semantic_role",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def _encode_batch(
    model: BGEM3FlagModel,
    texts: list[str],
    batch_size: int,
) -> tuple[list[list[float]], list[SparseVector]]:
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_list: list[list[float]] = output["dense_vecs"].tolist()
    sparse_list = [
        SparseVector(
            indices=[int(k) for k in lw.keys()],
            values=[float(v) for v in lw.values()],
        )
        for lw in output["lexical_weights"]
    ]
    return dense_list, sparse_list


def _build_payload(chunk: dict) -> dict:
    """All fields except embed_text go into Qdrant payload."""
    return {k: v for k, v in chunk.items() if k != "embed_text"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc_dir", type=Path)
    parser.add_argument(
        "--batch-size", type=int, default=32, help="BGE-M3 encode batch size (default: 32)"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Delete and recreate collection before indexing"
    )
    args = parser.parse_args()

    src = args.doc_dir / "rag_chunks.jsonl"
    if not src.exists():
        print(f"ERROR: {src} not found — run pass1_contextual.py first", file=sys.stderr)
        sys.exit(1)

    chunks = [
        json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    print(f"Loaded {len(chunks)} chunks from {src}")

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "road_distress_documents")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    _ensure_collection(client, collection, args.recreate)

    print("Loading BGE-M3 (fp16)…")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("Model loaded. Starting encoding…")

    total = 0
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        texts = [c["embed_text"] for c in batch]
        dense_list, sparse_list = _encode_batch(model, texts, args.batch_size)

        points = [
            PointStruct(
                id=_chunk_id_to_point_id(chunk["chunk_id"]),
                vector={"dense": dense, "sparse": sparse},
                payload=_build_payload(chunk),
            )
            for chunk, dense, sparse in zip(batch, dense_list, sparse_list, strict=True)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)
        print(f"  Upserted {total}/{len(chunks)}")

    count = client.count(collection_name=collection).count
    print(f"\nDone. Collection '{collection}' now has {count} points total.")


if __name__ == "__main__":
    main()
