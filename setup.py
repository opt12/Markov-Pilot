from setuptools import setup, find_packages


setup(name='Markov-Pilot',
      version='0.1',
      description='A package of reinforcement learning environments for flight '
                  'control using the JSBSim flight dynamics model.'
                  'This package supports Multi-Agents to learn how to fly.'
                  'It supports the MADDPG algorithm.',
      url='https://github.com/opt12/Markov-Pilot',
      author='Gor-Ren',
      license='MIT',
      install_requires=[
            'numpy',
            'gym',
            'matplotlib',   #TODO: a lot is missing
            'pandas'
      ],
      packages=find_packages(),
      classifiers=[
            'License :: OSI Approved :: MIT License',
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      python_requires='>=3.6',
      include_package_data=True,
      zip_safe=False)
