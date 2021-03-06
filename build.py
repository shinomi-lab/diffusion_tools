import difftools.algebra
import difftools.diffusion.ic
import difftools.diffusion.multi
import difftools.maximization
import difftools.trial

from typing import Any, Dict

ext_modules = [
    cc.distutils_extension()
    for cc in [
        difftools.algebra.cc,
        difftools.diffusion.ic.cc,
        difftools.diffusion.multi._jit.cc,
        difftools.maximization.cc,
        difftools.trial.cc,
    ]
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
        }
    )
