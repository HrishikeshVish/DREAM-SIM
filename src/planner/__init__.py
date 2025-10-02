import os
from os.path import abspath, dirname

from omegaconf import OmegaConf

from .utils.tools import join_string_underscore, parse_git_sha

# Export project root directory
#os.environ["PROJECT_ROOT"] = dirname(dirname(dirname(abspath(__file__))))
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = os.path.abspath('/media/csuser/DATA/seneva_multi/')

# Disable Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Register custom resolver for `OmegaConf`
OmegaConf.register_new_resolver(
    "join_string_underscore",
    join_string_underscore,
    replace=False,
    use_cache=True,
)
OmegaConf.register_new_resolver(
    "parse_git_sha",
    parse_git_sha,
    replace=False,
    use_cache=True,
)
