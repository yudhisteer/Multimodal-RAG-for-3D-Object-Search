import json
import os
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbeddingGenerator:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def load_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        return self.model.encode(image)

    def process_image_folder(self, image_folder: str) -> Dict[str, np.ndarray]:
        embeddings = {}

        # Supported image formats
        valid_extensions = {".jpg", ".jpeg", ".png"}

        for filename in os.listdir(image_folder):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_path = os.path.join(image_folder, filename)
                try:
                    image = self.load_image(image_path)
                    if image is not None:
                        embedding = self.generate_embedding(image)
                        embeddings[filename] = embedding
                        print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        return embeddings

    def save_embeddings(
        self, embeddings: Dict[str, np.ndarray], save_path: str
    ) -> None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        serializable_embeddings = {k: v.tolist() for k, v in embeddings.items()}

        with open(save_path, "w") as f:
            json.dump(serializable_embeddings, f)
        print(f"Embeddings saved to {save_path}")

    def load_embeddings(self, load_path: str) -> Dict[str, np.ndarray]:
        with open(load_path, "r") as f:
            data = json.load(f)

        embeddings = {k: np.array(v) for k, v in data.items()}
        
        # Print size and other details of the loaded embeddings
        for filename, array in embeddings.items():
            print(f"Loaded embedding for {filename}: shape = {array.shape}, dtype = {array.dtype}")

        return embeddings


if __name__ == "__main__":
    image_folder = "output/water"
    save_path = "embeddings/water_embeddings.json"  # Full path where to save

    generator = ImageEmbeddingGenerator()
    # embeddings = generator.process_image_folder(image_folder)
    # generator.save_embeddings(embeddings, save_path)
    loaded_embeddings = generator.load_embeddings(save_path)
