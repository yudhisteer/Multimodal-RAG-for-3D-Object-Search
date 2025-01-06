import torch
import sys
import subprocess

try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
    
    # Get version strings
    pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
    version_str = "".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{pyt_version_str}"
    ])
    
    # Install iopath
    subprocess.check_call(['pip', 'install', 'iopath'])
    
    if sys.platform.startswith("linux"):
        print("Trying to install wheel for PyTorch3D")
        wheel_url = f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"
        subprocess.check_call([
            'pip', 'install', '--no-index', '--no-cache-dir', 'pytorch3d',
            '-f', wheel_url
        ])
        
        # Check if pytorch3d is installed
        pip_list = subprocess.check_output(['pip', 'freeze']).decode()
        need_pytorch3d = not any(line.startswith("pytorch3d==") for line in pip_list.split('\n'))
        
    if need_pytorch3d:
        print(f"Failed to find/install wheel for {version_str}")
        
if need_pytorch3d:
    print("Installing PyTorch3D from source")
    subprocess.check_call(['pip', 'install', 'ninja'])
    subprocess.check_call([
        'pip', 'install', 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
    ])