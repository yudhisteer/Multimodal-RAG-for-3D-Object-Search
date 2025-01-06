import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from PIL import Image
import os
from typing import Annotated
print(torch.__version__)
print(torch.version.cuda)
import matplotlib.pyplot as plt

class MeshTextureRender:
    def __init__(self, 
                 obj_path: Annotated[str, "Path to the 3D model file"],
                 image_size: Annotated[int, "Image size for rendering"] = 512,
                 dist: Annotated[float, "Distance from the camera to the object"] = 1,
                 elev: Annotated[float, "Elevation angle"] = 30, 
                 azim: Annotated[float, "Azimuth angle"] = 180, 
                 fov: Annotated[float, "Field of view"] = 30,
                 y_offset: Annotated[float, "Y offset for the camera"] = 0.0) -> None:
        
        self.obj_path = obj_path
        self.image_size = image_size
        self.device = self.get_device()
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.fov = fov
        self.y_offset = y_offset


    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device("cuda:0")
        else:
            print("Using CPU")
            return torch.device("cpu")
        
    def load_mesh(self) -> Meshes:
        try:
            mesh = load_objs_as_meshes([self.obj_path], device=self.device, load_textures=True)
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return None
        return mesh
    
    def renderer(self) -> MeshRenderer:
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim)
        
        # Adjust the Y translation
        T[..., 1] += self.y_offset
        
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self.fov)
        raster_settings = RasterizationSettings(image_size=self.image_size, blur_radius=0.0, faces_per_pixel=1)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings), shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights))
        return renderer


    def view_3d_model(self) -> None:
        mesh = self.load_mesh()
        renderer = self.renderer()
        
        # Render the textured mesh
        images = renderer(mesh)
        
        # Convert the rendered image to numpy and display
        image = images[0, ..., :3].cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def render_360_views(self, num_views: int = 10, output_prefix: str = "render") -> str:
        mesh = self.load_mesh()
        
        # Create output directory based on obj filename
        obj_name = os.path.splitext(os.path.basename(self.obj_path))[0]
        output_dir = os.path.join("output", obj_name)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_views):
            angle = i * 360 / num_views
            # Update azimuth for each view
            self.azim = angle
            # Get renderer with updated azimuth
            renderer = self.renderer()
            image = renderer(mesh)

            # Convert the rendered image to numpy
            image = image[0, ..., :3].cpu().numpy()
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            output_path = os.path.join(output_dir, f"{output_prefix}_{i+1:03d}.png")
            pil_image.save(output_path)
            print(f"Saved image {i+1} to {output_path}")

        print(f"{num_views} PNG images have been saved to: {output_dir}")
        # return output_dir







if __name__ == "__main__":

    # Usage
    #render_360_views("models/water/water.obj", num_views=10, image_size=512, output_prefix="water")

    # view_3d_model("models/water/water.obj")

    configs = {
        "dist": 3.8,
        "elev": 30,
        "azim": 180,
        "fov": 5,
        "image_size": 512,
        "y_offset": 0.1
    }

    mesh_texture_render = MeshTextureRender("models/basket/basket.obj", **configs)
    mesh_texture_render.view_3d_model()
    #mesh_texture_render.render_360_views(num_views=10,  output_prefix="basket")


