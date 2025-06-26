"""To adapt musicfm code into samantha, there are several
things we need:

1. musicfm is using webdataset saved on bytenas, and the
   paths are hard-coded so we need to change the behavior
   of glob in order to find path in other places.
2. musicfm yaml conf can only be run from the parent folder
   of recipes (since there are hard-coded relative paths)
   so we need to change the working directory.
3. Both musicfm and samantha have a copy of `musiclm`, but
   they both changed musiclm. We need to keep both versions
   and load the correct version when running musicfm. Below
   we use an import hook to override imported musiclm submodule
   path to the musicfm version.
"""
import os
import sys
import importlib.machinery
import importlib.util
from glob import iglob


## 1
from recipes.datasets.mir.base import hdfs_ls
def _glob(x, recursive=False):
    old_prefix = "/mnt/bn/audio-diffusion/"
    if old_prefix not in x:
        return iglob(x, recursive=recursive)
    new_prefix = "/home/byte_speech_sv/data/"
    hdfs_dir = new_prefix + x[len(old_prefix):]
    hdfs_paths = hdfs_ls(hdfs_dir)
    ## For debug, fetch only 1 path since the loading is slow
    # hdfs_paths = [hdfs_paths[0]] if len(hdfs_paths) > 0 else []
    return [f"pipe:hdfs dfs -cat {p}" for p in hdfs_paths]
def convert_nas_to_hdfs_in_glob():
    import glob as original_glob_module
    original_glob_module.glob = _glob

## 2
curr_dir = os.path.dirname(__file__)
samantha_root_dir = os.path.normpath(os.path.join(curr_dir, "../../../"))
extra_recipes_dir = os.path.normpath(os.path.join(curr_dir, "../deps/"))
os.chdir(samantha_root_dir)

## 3
class MusicLMImporter:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "recipes.musiclm":
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
                spec.submodule_search_locations = [
                    os.path.join(
                        extra_recipes_dir,
                        os.path.relpath(
                            spec.submodule_search_locations[0],
                            samantha_root_dir,
                        )
                    )
                ]
            finally:
                sys.meta_path.insert(0, self)
            return spec
        return None


if not any(isinstance(hook, MusicLMImporter) for hook in sys.meta_path):
    sys.meta_path.insert(0, MusicLMImporter())