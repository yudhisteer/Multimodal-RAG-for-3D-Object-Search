import json
import os
from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class MetadataEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, metadata: Dict[str, Any]) -> np.ndarray:
        structured_input = json.dumps(metadata)
        embedding = self.model.encode(structured_input)
        return embedding

    def save_embedding(self, embedding: np.ndarray, save_path: str):
        embedding_list = embedding.tolist()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"embedding": embedding_list}, f)

    def process_metadata(self, metadata_path: str, output_path: str) -> None:
        try:
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Generate embedding
            embedding = self.generate_embedding(metadata)

            # Save embedding
            self.save_embedding(embedding, output_path)
            print(f"Successfully saved embedding to {output_path}")

        except Exception as e:
            print(f"Error processing metadata: {str(e)}")


if __name__ == "__main__":
    embedder = MetadataEmbedder()

    metadata_path = "metadata/basket_metadata.json"
    output_path = "embeddings/basket_metadata_embeddings.json"

    embedder.process_metadata(metadata_path, output_path)
