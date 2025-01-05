import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from PIL import Image
import os


def render_360_views(obj_path, num_views=10, image_size=512, output_prefix="render"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the mesh
    try:
        mesh = load_objs_as_meshes([obj_path], device=device)
    except Exception as e:
        print(f"Error loading mesh from {obj_path}: {e}")
        return None

    # Define the renderer
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )
    
    # Place a point light in front of the object
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, lights=lights)
    )

    # Create output directory based on obj filename
    obj_name = os.path.splitext(os.path.basename(obj_path))[0]
    output_dir = os.path.join("output", obj_name)
    os.makedirs(output_dir, exist_ok=True)

    # Render images
    for i in range(num_views):
        angle = i * 360 / num_views
        R, T = look_at_view_transform(dist=0.7, elev=10, azim=angle)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        image = renderer(mesh, cameras=cameras)
        image = image[0, ..., :3].cpu().numpy()  # RGB only
        
        # Convert to PIL Image and save as PNG
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Save to the new directory
        output_path = os.path.join(output_dir, f"{output_prefix}_{i+1:03d}.png")
        pil_image.save(output_path)

    print(f"{num_views} PNG images have been saved to: {output_dir}")

# Usage
render_360_views("models/water/water.obj", num_views=10, image_size=512, output_prefix="model_view")
