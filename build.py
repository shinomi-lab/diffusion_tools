import difftools.algebra
import difftools.diffusion
import difftools.maximization

from typing import Any, Dict

ext_modules = [
    cc.distutils_extension()
    for cc in [
        difftools.algebra.cc,
        difftools.diffusion.cc,
        difftools.maximization.cc,
    ]
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
        }
    )
