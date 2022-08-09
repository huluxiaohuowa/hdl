from setuptools import setup, find_packages
import versioneer

setup(
    name="xdl",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Jianxing Hu",
    author_email="jianxing.hu@xtalpi.com",
    description="DL framework from Jianxing",
    url="", 
    packages=find_packages(), 
    python_requires='>=3',
    data_files=[
        (
            'data_file', 
            [
                'hdl/features/vocab.txt'
            ]
        ),
    ],
    include_package_data=True
)