from src.views import MeshTextureRender

if __name__ == "__main__":
    configs = {
        "dist": 3.8,
        "elev": 30,
        "azim": 180,
        "fov": 5,
        "image_size": 512,
        "y_offset": -0.1,
    }

    mesh_texture_render = MeshTextureRender("models/water/water.obj", **configs)
    # mesh_texture_render.view_3d_model()
    mesh_texture_render.render_360_views(num_views=10, output_prefix="water")
