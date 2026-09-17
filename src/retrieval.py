from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_retrieval_config(retrieval_data_dir):
    retrieval_data_dir = Path(retrieval_data_dir)

    with open(
        retrieval_data_dir / "config.json",
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    return config


def load_retrieval_data(retrieval_data_dir):
    retrieval_data_dir = Path(retrieval_data_dir)

    chunks_df = pd.read_parquet(
        retrieval_data_dir / "chunks.parquet"
    )

    document_embeddings = np.load(
        retrieval_data_dir / "embeddings.npy"
    )

    if len(chunks_df) != document_embeddings.shape[0]:
        raise ValueError(
            "Number of chunks does not match "
            "number of document embeddings."
        )

    return chunks_df, document_embeddings


def load_embedding_model(
    model_name,
    device=None,
):
    model = SentenceTransformer(
        model_name,
        device=device,
    )

    return model


def encode_query(
    query,
    embedding_model,
    query_prefix,
):
    query_text = (
        query_prefix
        + query
    )

    query_embedding = embedding_model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return query_embedding


def dense_search(
    query,
    chunks_df,
    document_embeddings,
    embedding_model,
    query_prefix,
    top_k=5,
):
    query_embedding = encode_query(
        query=query,
        embedding_model=embedding_model,
        query_prefix=query_prefix,
    )

    scores = (
        document_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = (
        chunks_df
        .iloc[top_indices]
        .copy()
    )

    results.insert(
        0,
        "score",
        scores[top_indices],
    )

    result_columns = [
        "score",
        "chunk_id",
        "document_id",
        "source",
        "document_title",
        "year",
        "section_path",
        "page",
        "pdf_page",
        "text",
    ]

    existing_columns = [
        column
        for column in result_columns
        if column in results.columns
    ]

    return (
        results[
            existing_columns
        ]
        .reset_index(drop=True)
    )