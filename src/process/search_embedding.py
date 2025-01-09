import os
import json
from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any


class ProductSearcher:
    def __init__(
        self,
        base_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        visual_weight: float = 0.6,
        metadata_weight: float = 0.4,
        clip_model: str = "clip-ViT-B-32",
    ):
        """
        Initialize ProductSearcher.

        Args:
            base_dir: Base directory containing all embedding folders
            (The directory that contains image_embeddings, metadata, metadata_embeddings folders)
        """
        self.base_dir = base_dir
        self.model = SentenceTransformer(model_name)
        self.visual_weight = visual_weight
        self.metadata_weight = metadata_weight
        self.clip_model = SentenceTransformer(clip_model)

        # Define paths based on your structure
        self.image_embeddings_dir = os.path.join(base_dir, "image_embeddings")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.metadata_embeddings_dir = os.path.join(base_dir, "metadata_embeddings")

        # Load all embeddings
        self.products_data = self.load_all_embeddings()

    def load_all_embeddings(self) -> Dict[str, Dict]:
        """Load embeddings from all folders."""
        products_data = {}

        # Get list of products from metadata folder
        for metadata_file in os.listdir(self.metadata_dir):
            if not metadata_file.endswith(".json"):
                continue

            product_name = metadata_file.replace("_metadata.json", "")

            try:
                # Load metadata
                with open(os.path.join(self.metadata_dir, metadata_file), "r") as f:
                    metadata = json.load(f)

                # Load image embeddings
                image_emb_file = f"{product_name}_image_embeddings.json"
                with open(
                    os.path.join(self.image_embeddings_dir, image_emb_file), "r"
                ) as f:
                    visual_embeddings = json.load(f)

                # Load metadata embeddings
                meta_emb_file = f"{product_name}_metadata_embeddings.json"
                with open(
                    os.path.join(self.metadata_embeddings_dir, meta_emb_file), "r"
                ) as f:
                    metadata_embedding = json.load(f)

                # Store in products_data
                products_data[product_name] = {
                    "metadata": metadata,
                    "metadata_embedding": np.array(metadata_embedding["embedding"]),
                    "visual_embeddings": {
                        img_path: np.array(emb)
                        for img_path, emb in visual_embeddings.items()
                    },
                }

            except Exception as e:
                print(f"Error loading data for {product_name}: {str(e)}")
                continue

        # save products_data text
        # with open(os.path.join(self.base_dir, 'products_data.txt'), 'w') as f:
        #     f.write(str(products_data))

        return products_data

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query)

    def compute_metadata_similarity(
        self, query_embedding: np.ndarray, product_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity for metadata embeddings."""
        return cosine_similarity(
            query_embedding.reshape(1, -1), product_embedding.reshape(1, -1)
        )[0][0]

    def compute_visual_similarity(self, query: str, visual_embeddings: Dict[str, np.ndarray]) -> float:
        """Compute average cosine similarity across all image embeddings."""
        # Use CLIP to create query embedding (passing the raw query text, not an embedding)
        query_embedding = self.clip_model.encode(query)
        
        similarities = []
        for img_embedding in visual_embeddings.values():
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                img_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        return np.mean(similarities)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for products matching the query using both visual and metadata embeddings.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of top matching products with combined scores
        """
        # For metadata comparison
        metadata_query_embedding = self.model.encode(query)
        
        results = []
        for sku, data in self.products_data.items():
            metadata_sim = self.compute_metadata_similarity(
                metadata_query_embedding, 
                data['metadata_embedding']
            )
            
            visual_sim = self.compute_visual_similarity(
                query,
                data['visual_embeddings']
            )
            # Calculate weighted combined score
            combined_score = (
                self.visual_weight * visual_sim + self.metadata_weight * metadata_sim
            )

            results.append(
                {
                    "sku": sku,
                    "metadata": data["metadata"],
                    "scores": {
                        "visual_similarity": float(visual_sim),
                        "metadata_similarity": float(metadata_sim),
                        "combined_score": float(combined_score),
                    },
                }
            )

        # Sort by combined score
        results.sort(key=lambda x: x["scores"]["combined_score"], reverse=True)

        # Return top k results
        return results[:top_k]


if __name__ == "__main__":

    searcher = ProductSearcher(os.getcwd(), visual_weight=0.6, metadata_weight=0.4)

    # Print available keys first
    print("Available products:", list(searcher.products_data.keys()))

    query = "a green vintage kettle"
    query_embedding = searcher.embed_query(query)

    ###----Metadata Similarity----###
    first_product_key = list(searcher.products_data.keys())[0]  # Get first product key
    metadata_similarity = searcher.compute_metadata_similarity(
        query_embedding, searcher.products_data[first_product_key]["metadata_embedding"]
    )
    print(f"Metadata similarity for {first_product_key}: {metadata_similarity}")

    second_product_key = list(searcher.products_data.keys())[1]  # Get second product key
    metadata_similarity = searcher.compute_metadata_similarity(
        query_embedding,
        searcher.products_data[second_product_key]["metadata_embedding"],
    )
    print(f"Metadata similarity for {second_product_key}: {metadata_similarity}")

    ###----Visual Similarity----###
    visual_similarity = searcher.compute_visual_similarity(
        query, searcher.products_data[first_product_key]["visual_embeddings"]
    )
    print(f"Visual similarity for {first_product_key}: {visual_similarity}")

    visual_similarity = searcher.compute_visual_similarity(
        query, searcher.products_data[second_product_key]["visual_embeddings"]
    )
    print(f"Visual similarity for {second_product_key}: {visual_similarity}")

    ###----Search----###
    results = searcher.search(query, top_k=3)

    for idx, result in enumerate(results, 1):
        print(f"\nMatch {idx}:")
        print(f"SKU: {result['sku']}")
        print(f"Scores:")
        print(f"  Visual Similarity: {result['scores']['visual_similarity']:.3f}")
        print(f"  Metadata Similarity: {result['scores']['metadata_similarity']:.3f}")
        print(f"  Combined Score: {result['scores']['combined_score']:.3f}")
        print(f"Image Paths: {result['metadata']['image_paths']}")
