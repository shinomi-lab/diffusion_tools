import difftools.diffusion as dd
import difftools.maximization as dm

from typing import Any, Dict

ext_modules = [dd.cc.distutils_extension(), dm.cc.distutils_extension()]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
        }
    )
