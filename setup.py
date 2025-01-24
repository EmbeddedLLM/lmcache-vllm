from setuptools import setup, find_packages
import torch
import os

def _is_hip():
    if torch.cuda.is_available() and torch.version.hip:
        return True
    else:
        return False

installed_dependencies = [
        "lmcache>=0.1.4",
]

if not _is_hip():
    installed_dependencies.append([
        "vllm==0.6.2",
    ])
else:
    installed_dependencies.append([
        "vllm==0.6.2+rocm634",
    ])
    

setup(
    name="lmcache_vllm",
    version="0.6.2.3",
    description = "lmcache_vllm: LMCache's wrapper for vllm",
    author = "LMCache team",
    author_email = "lmcacheteam@gmail.com",
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=installed_dependencies,
    entry_points={
        'console_scripts': [
            "lmcache_vllm=lmcache_vllm.script:main"
        ],
    },
)

