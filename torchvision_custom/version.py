__version__ = '0.10.0+cu111'
git_version = 'ae9963fd077619c7d2a134813e35551943e87458'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
 