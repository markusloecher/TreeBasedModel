from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='SelectiveHierarchicalShrinkage',
      version="0.1",
      description="Accompanining package for master thesis",
      packages=find_packages(),
      install_requires=requirements,
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/HS_performance_rf-run'],
      zip_safe=False)
