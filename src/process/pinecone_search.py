import os
import json
from typing import Dict, List, Any
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


load_dotenv()

class PineconeProductSearcher:
    def __init__(
        self,
        metadata_index_name: str,
        visual_index_name: str,
        model_name: str = "all-MiniLM-L6-v2",
        visual_weight: float = 0.6,
        metadata_weight: float = 0.4,
        clip_model: str = "clip-ViT-B-32-multilingual-v1",
    ):
        """
        Initialize PineconeProductSearcher with separate indices.
        
        Args:
            metadata_index_name: Name of the Pinecone index for metadata
            visual_index_name: Name of the Pinecone index for visual embeddings
        """
        # Initialize models
        self.model = SentenceTransformer(model_name)
        self.clip_model = SentenceTransformer(clip_model)
        self.visual_weight = visual_weight
        self.metadata_weight = metadata_weight
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get dimensions
        self.metadata_dim = self.model.get_sentence_embedding_dimension()
        # if self.clip_model == "clip-ViT-B-32":
        #     self.visual_dim = 512
        # else:
        #     
        # 
        self.visual_dim = self.clip_model.get_sentence_embedding_dimension()

        print(f"Metadata dimension: {self.metadata_dim}")
        print(f"Visual dimension: {self.visual_dim}")
        
        # Create or get indices
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")

        
        # Setup metadata index
        if metadata_index_name not in existing_indexes:
            self.pc.create_index(
                name=metadata_index_name,
                dimension=self.metadata_dim,
                metadata_config={
                    "indexed": ["sku", "product_name"]
                }
            )
        self.metadata_index = self.pc.Index(metadata_index_name)
        
        # Setup visual index
        if visual_index_name not in existing_indexes:
            self.pc.create_index(
                name=visual_index_name,
                dimension=self.visual_dim,
                metadata_config={
                    "indexed": ["sku", "product_name", "image_path"]
                }
            )
        self.visual_index = self.pc.Index(visual_index_name)

    def clean_indices(self):
        """Clean all data from both indices."""
        print("Cleaning indices...")
        try:
            # Check metadata index stats
            metadata_stats = self.metadata_index.describe_index_stats()
            if metadata_stats.total_vector_count > 0:
                self.metadata_index.delete(
                    delete_all=True,
                    namespace=''  # Specify default namespace
                )
                print("Metadata index cleaned")
            else:
                print("Metadata index already empty")
        except Exception as e:
            print(f"Warning: Could not clean metadata index: {str(e)}")

        try:
            # Check visual index stats
            visual_stats = self.visual_index.describe_index_stats()
            if visual_stats.total_vector_count > 0:
                self.visual_index.delete(
                    delete_all=True,
                    namespace=''  # Specify default namespace
                )
                print("Visual index cleaned")
            else:
                print("Visual index already empty")
        except Exception as e:
            print(f"Warning: Could not clean visual index: {str(e)}")

        print("Clean operation completed")

    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """Flatten nested metadata to simple key-value pairs."""
        flat_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat_metadata[key] = value
            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                flat_metadata[key] = value
            elif isinstance(value, dict):
                # For nested dictionaries, create flattened keys
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        flat_metadata[f"{key}_{sub_key}"] = sub_value
        return flat_metadata

    def load_and_store_embeddings(self, base_dir: str, batch_size: int = 100) -> None:
        """Load embeddings and store in separate indices."""
        image_embeddings_dir = os.path.join(base_dir, "image_embeddings")
        metadata_dir = os.path.join(base_dir, "metadata")
        metadata_embeddings_dir = os.path.join(base_dir, "metadata_embeddings")
        
        metadata_batch = []
        visual_batch = []
        
        for metadata_file in os.listdir(metadata_dir):
            if not metadata_file.endswith(".json"):
                continue
                
            product_name = metadata_file.replace("_metadata.json", "")
            
            try:
                # Load metadata
                with open(os.path.join(metadata_dir, metadata_file), "r") as f:
                    metadata = json.load(f)
                    sku = metadata.get('sku')
                    if not sku:
                        print(f"No SKU found in metadata for {product_name}")
                        continue
                
                # Flatten metadata
                flat_metadata = self._flatten_metadata(metadata)
                flat_metadata["product_name"] = product_name
                flat_metadata["sku"] = sku
                
                # Load metadata embeddings
                meta_emb_file = f"{product_name}_metadata_embeddings.json"
                with open(os.path.join(metadata_embeddings_dir, meta_emb_file), "r") as f:
                    metadata_embedding = json.load(f)
                
                # Add to metadata batch
                metadata_batch.append((
                    f"{sku}_metadata",
                    metadata_embedding["embedding"],
                    flat_metadata
                ))
                
                # Load and store image embeddings
                image_emb_file = f"{product_name}_image_embeddings.json"
                with open(os.path.join(image_embeddings_dir, image_emb_file), "r") as f:
                    visual_embeddings = json.load(f)
                    
                for img_path, embedding in visual_embeddings.items():
                    visual_batch.append((
                        f"{sku}_visual_{img_path}",
                        embedding,
                        {
                            **flat_metadata,
                            "image_path": img_path
                        }
                    ))
                
                # Batch upsert if either batch is full
                if len(metadata_batch) >= batch_size:
                    self.metadata_index.upsert(vectors=metadata_batch)
                    metadata_batch = []
                
                if len(visual_batch) >= batch_size:
                    self.visual_index.upsert(vectors=visual_batch)
                    visual_batch = []

                # print the number of vectors upserted
                print(f"Upserted {len(metadata_batch)} metadata vectors and {len(visual_batch)} visual vectors")
            except Exception as e:
                print(f"Error processing {product_name}: {str(e)}")
                continue
        
        # Upsert any remaining vectors
        if metadata_batch:
            self.metadata_index.upsert(vectors=metadata_batch)
        if visual_batch:
            self.visual_index.upsert(vectors=visual_batch)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search using both indices."""
        # Get query embeddings
        metadata_query = self.model.encode(query)
        visual_query = self.clip_model.encode(query)
        
        # Query both indices
        metadata_results = self.metadata_index.query(
            vector=metadata_query.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
        
        visual_results = self.visual_index.query(
            vector=visual_query.tolist(),
            top_k=top_k * 3,  # Get more visual results since we have multiple images per product
            include_metadata=True,
            include_values=True
        )
        
        # Process and combine results using cosine similarity
        sku_scores = {}
        
        # Process metadata results
        for match in metadata_results.matches:
            sku = match.metadata["sku"]
            metadata_vector = np.array(match.values).reshape(1, -1)
            metadata_sim = cosine_similarity(
                metadata_query.reshape(1, -1),
                metadata_vector
            )[0][0]
            
            sku_scores[sku] = {
                "metadata": match.metadata,
                "metadata_similarity": metadata_sim,
                "visual_similarity": 0,
                "sku": sku
            }
        
        # Process visual results
        visual_sims = {}
        for match in visual_results.matches:
            sku = match.metadata["sku"]
            visual_vector = np.array(match.values).reshape(1, -1)
            visual_sim = cosine_similarity(
                visual_query.reshape(1, -1),
                visual_vector
            )[0][0]
            
            if sku not in visual_sims:
                visual_sims[sku] = []
            visual_sims[sku].append(visual_sim)
        
        # Average visual similarities
        for sku, similarities in visual_sims.items():
            avg_visual_sim = np.mean(similarities)
            if sku not in sku_scores:
                sku_scores[sku] = {
                    "metadata": visual_results.matches[0].metadata,
                    "metadata_similarity": 0,
                    "visual_similarity": avg_visual_sim,
                    "sku": sku
                }
            else:
                sku_scores[sku]["visual_similarity"] = avg_visual_sim
        
        # Calculate combined scores
        results = []
        for sku, data in sku_scores.items():
            combined_score = (
                self.visual_weight * data["visual_similarity"] +
                self.metadata_weight * data["metadata_similarity"]
            )
            
            results.append({
                "sku": sku,
                "metadata": data["metadata"],
                "scores": {
                    "visual_similarity": float(data["visual_similarity"]),
                    "metadata_similarity": float(data["metadata_similarity"]),
                    "combined_score": float(combined_score)
                }
            })
        
        # Sort by combined score and return top_k
        results.sort(key=lambda x: x["scores"]["combined_score"], reverse=True)
        return results[:top_k]

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results in a readable way."""
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\nResult #{i}")
            output.append("=" * 30)
            
            # Basic info
            output.append(f"SKU: {result['sku']}")
            output.append(f"Product: {result['metadata'].get('product_name', 'N/A')}")
            
            # Scores section
            scores = result['scores']
            output.append("\nScores:")
            output.append(f"  Combined Score:       {scores['combined_score']:.3f}")
            output.append(f"  Visual Similarity:    {scores['visual_similarity']:.3f}")
            output.append(f"  Metadata Similarity:  {scores['metadata_similarity']:.3f}")

        return "\n".join(output)

if __name__ == "__main__":
    searcher = PineconeProductSearcher(
        metadata_index_name="metadata",
        visual_index_name="image"
    )
    
    # clean indices
    #searcher.clean_indices()
    
    #searcher.load_and_store_embeddings(base_dir=os.getcwd())
    results = searcher.search("a green vintage kettle", top_k=1)
    print(searcher._format_results(results))