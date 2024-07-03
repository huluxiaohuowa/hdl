from setuptools import setup, find_packages
import setuptools_scm


def read_requirements():
    """读取 requirements.txt 文件并返回依赖列表"""
    with open('requirements.txt', 'r', encoding='utf-8') as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith('#')
        ]


def custom_version_scheme(version):
    """自定义版本号方案，确保没有 .dev 后缀"""
    if version.exact:
        return version.format_with("{tag}")
    elif version.distance:
        return f"{version.format_next_version()}.post{version.distance}"
    else:
        return version.format_with("0.0.0")


def custom_local_scheme(version):
    """自定义本地版本方案，确保没有本地版本后缀"""
    return ""


setup(
    name="hjxdl",
    use_scm_version={
        "version_scheme": custom_version_scheme,
        "local_scheme": custom_local_scheme,
        "write_to": "hdl/_version.py"
    },
    author="Jianxing Hu",
    author_email="j.hu@pku.edu.cn",
    description="A collection of functions for Jupyter notebooks",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/huluxiaohuowa/hdl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['setuptools_scm'],
    install_requires=read_requirements()
)
