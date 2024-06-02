from setuptools import setup, find_packages

setup(
    name='cv_utils',
    version='0.1.0',
    description='A utility library for computer vision',
    author='Sehyun Cha',
    author_email='cshyundev@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/cshyundev/cv_utils",
    packages=find_packages(where='src'),  
    package_dir={'': 'src'}, 
    install_requires=[
        'numpy',
        'opencv-python',
        # 'pycolmap',
        'absl-py',
        'tifffile',
        'open3d',
        'scikit-image',
        'matplotlib',
        'torch'
    ],
    python_requires='>=3.6',
)