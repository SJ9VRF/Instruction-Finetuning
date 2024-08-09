from setuptools import setup, find_packages

setup(
    name='legal_summarization',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Fine-tuning Meta-Llama-3-8B-Instruct for Legal Summarization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourgithub/legal_summarization',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'peft',
        'trl',
        'accelerate',
        'bitsandbytes',
        'datasets'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'train_model=legal_summarization.train_model:main',
            'inference=legal_summarization.inference:main'
        ],
    }
)

