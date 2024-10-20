from setuptools import setup, find_packages
setup(
    name='colabdesign',
    version='1.1.1',
    description='Making Protein Design accessible to all via Google Colab!',
    long_description="Making Protein Design accessible to all via Google Colab!",
    long_description_content_type='text/markdown',
    packages=find_packages(include=['colabdesign*']),
    install_requires=['py3Dmol','absl-py','biopython',
                      'chex','dm-haiku','dm-tree',
                      'immutabledict','ml-collections',
                      'numpy','pandas','scipy==1.12.0','optax','joblib',
                      'matplotlib'],
    include_package_data=True
)
