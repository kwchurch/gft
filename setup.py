import os
import setuptools

long_description = "GFT (general fine-tuning) A Little Language for Deepnets: " \
                   "1-line programs for fine-tuning, inference and more"

# with open("requirements/requirements_without_adapters.txt") as fin:
#     REQUIRED_PACKAGES = fin.read()

with open("requirements/requirements_without_adapters-cpu.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

def get_package_data_files(package, data, package_dir=None):
    """
    Helps to list all specified files in package including files in directories
    since `package_data` ignores directories.
    """
    if package_dir is None:
        package_dir = os.path.join(*package.split('.'))
    all_files = []
    for f in data:
        path = os.path.join(package_dir, f)
        if os.path.isfile(path):
            all_files.append(f)
            continue
        for root, _dirs, files in os.walk(path, followlinks=True):
            root = os.path.relpath(root, package_dir)
            for file in files:
                file = os.path.join(root, file)
                if file not in all_files:
                    all_files.append(file)
    return all_files


setuptools.setup(
    name="gft-cpu",
    version="0.1.5",
    author="GFT Authors",
    author_email="kenneth.ward.church@gmail.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/kwchurch/gft",
    packages=setuptools.find_packages(
        where='.', exclude=('examples*', 'tests*', 'doc*', 'requirements*', 'datasets*')),
    # package_data={
    #     'paddlenlp.transformers.layoutxlm': get_package_data_files(
    #         'paddlenlp.transformers.layoutxlm', ['visual_backbone.yaml']),
    # },
    setup_requires=['cython', 'numpy'],
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'gft_summary=gft.gft_summary:main',
            'gft_predict=gft.gft_predict:main',
            'gft_fit=gft.gft_fit:main',
            'gft_eval=gft.gft_eval:main',
            'gft_cat_data=gft.gft_cat_data:main',
        ]
    },
    license='Apache 2.0')

# setuptools.setup(
#     name="gft",
#     version="0.1.4",
#     author="GFT Authors",
#     author_email="kenneth.ward.church@gmail.com",
#     description=long_description,
#     long_description=long_description,
#     long_description_content_type="text/plain",
#     url="https://github.com/kwchurch/gft",
#     packages=setuptools.find_packages(
#         where='.', exclude=('examples*', 'tests*', 'doc*', 'requirements*', 'datasets*')),
#     # package_data={
#     #     'paddlenlp.transformers.layoutxlm': get_package_data_files(
#     #         'paddlenlp.transformers.layoutxlm', ['visual_backbone.yaml']),
#     # },
#     setup_requires=['cython', 'numpy'],
#     install_requires=REQUIRED_PACKAGES,
#     python_requires='>=3.6',
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: Python :: 3.7',
#         'Programming Language :: Python :: 3.8',
#         'License :: OSI Approved :: Apache Software License',
#         'Operating System :: OS Independent',
#     ],
#     entry_points={
#         'console_scripts': [
#             'gft_summary=gft.gft_summary:main',
#             'gft_predict=gft.gft_predict:main',
#             'gft_fit=gft.gft_fit:main',
#             'gft_eval=gft.gft_eval:main',
#             'gft_cat_data=gft.gft_cat_data:main',
#         ]
#     },
#     license='Apache 2.0')
