# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['difftools']

package_data = \
{'': ['*']}

install_requires = \
['joblib', 'matplotlib', 'networkx', 'numba', 'numpy', 'scipy']

setup_kwargs = {
    'name': 'diffusion-tools',
    'version': '0.2.0',
    'description': '',
    'long_description': None,
    'author': 'Shinomiya-Lab',
    'author_email': 'lab.shinomiya@outlook.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
