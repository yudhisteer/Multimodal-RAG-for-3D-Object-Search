from typing import Dict, List, Any
import os
import random
from PIL import Image
from io import BytesIO
import base64
import json
import ollama
import string


class MetadataGenerator:
    def __init__(
        self,
        prompt_template: str,
        analysis_prompt: str,
        structuring_prompt: str,
        model_name: str = "llava",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.analysis_prompt = analysis_prompt
        self.structuring_prompt = structuring_prompt

    def select_random_images(self, image_folder: str, num_images: int = 3) -> List[str]:
        valid_extensions = {".jpg", ".jpeg", ".png"}
        all_images = [
            f
            for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        selected_images = random.sample(all_images, min(num_images, len(all_images)))
        return [os.path.join(image_folder, img) for img in selected_images]

    def image_to_base64(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_metadata(self, image_paths: List[str]) -> Dict[str, Any]:
        try:
            combined_metadata = {}

            for image_path in image_paths:
                base64_image = self.image_to_base64(image_path)

                # Step 1: Initial Analysis
                analysis_response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.analysis_prompt,
                            "images": [base64_image],
                        }
                    ],
                )

                initial_analysis = analysis_response["message"]["content"]
                print(f"Initial Analysis for {image_path}:")
                print(initial_analysis)

                # Step 2: Structure the Analysis
                structure_response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.analysis_prompt,
                            "images": [base64_image],
                        },
                        {"role": "assistant", "content": initial_analysis},
                        {"role": "user", "content": self.structuring_prompt},
                    ],
                )

                try:
                    cleaned_response = self.clean_json_response(
                        structure_response["message"]["content"]
                    )
                    metadata = json.loads(cleaned_response)
                    # Merge with existing metadata if not empty
                    if not combined_metadata:
                        combined_metadata = metadata
                    else:
                        self.merge_metadata(combined_metadata, metadata)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from response for {image_path}: {e}")
                    print("Response content:", structure_response["message"]["content"])

            return combined_metadata

        except Exception as e:
            print(f"Error generating metadata: {str(e)}")
            return {}

    def merge_metadata(self, existing: Dict, new: Dict) -> None:
        for key in new:
            if isinstance(new[key], list):
                if key in existing:
                    existing[key].extend(x for x in new[key] if x not in existing[key])
            elif isinstance(new[key], dict):
                if key not in existing:
                    existing[key] = {}
                self.merge_metadata(existing[key], new[key])
            else:
                # For simple values, keep the existing unless empty
                if key not in existing or not existing[key]:
                    existing[key] = new[key]

    def clean_json_response(self, response_text: str) -> str:
        """Clean response text to extract valid JSON."""
        # If response contains markdown JSON blocks
        if "```json" in response_text:
            # Extract content between ```json and ```
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end]

        # Remove any leading/trailing whitespace
        response_text = response_text.strip()

        # Ensure it starts with {
        if not response_text.startswith("{"):
            response_text = response_text[response_text.find("{") :]

        # Ensure it ends with }
        if not response_text.endswith("}"):
            response_text = response_text[: response_text.rfind("}") + 1]

        return response_text

    def save_metadata(self, metadata: Dict, save_path: str) -> None:
        try:
            with open(save_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to {save_path}")
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")

    def generate_sku(self) -> str:
        """Generate random 8-character SKU with letters and numbers."""
        # Characters to choose from (letters and numbers)
        characters = string.ascii_uppercase + string.digits
        # Generate 8 random characters
        sku = "".join(random.choice(characters) for _ in range(8))
        return sku

    def process_product_folder(self, folder_path: str, output_path: str) -> Dict:
        try:
            # Generate random SKU
            product_sku = self.generate_sku()

            # Select random images
            selected_images = self.select_random_images(folder_path)

            # Generate metadata
            metadata = self.generate_metadata(selected_images)

            # Add SKU to metadata
            metadata["sku"] = product_sku

            # Add image paths to metadata
            metadata["image_paths"] = selected_images

            # Save metadata if output path provided
            if output_path:
                self.save_metadata(metadata, output_path)

            return metadata

        except Exception as e:
            print(f"Error processing folder {folder_path}: {str(e)}")
            return {}


if __name__ == "__main__":
    prompt_template = """Analyze this furniture/product image in detail. Create a comprehensive JSON metadata structure that includes all attributes you can identify. Be specific and detailed in your analysis. Make sure the output is in valid JSON format. Important: Include any attributes you find relevant for this specific item, don't limit yourself to standard categories."""
    analysis_prompt = """Analyze this image and output ONLY valid JSON. Do not include any other text, markdown formatting, or explanations. Start directly with { and end with }."""
    structuring_prompt = """Based on the previous analysis, create a structured JSON metadata that organizes all these attributes effectively. Ensure the output is in valid JSON format."""

    generator = MetadataGenerator(prompt_template, analysis_prompt, structuring_prompt)

    # Process a product folder
    folder_path = "output/basket"
    output_path = "metadata/basket_metadata.json"

    metadata = generator.process_product_folder(folder_path, output_path)
    print("Generated metadata:", json.dumps(metadata, indent=2))
