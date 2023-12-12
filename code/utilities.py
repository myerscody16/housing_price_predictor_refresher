import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_preprocessing(path_to_notebook):
    with open(path_to_notebook, 'r', encoding='utf-8') as notebook:
        notebook_content = nbformat.read(notebook, as_version=4)
    execute_preprocessor = ExecutePreprocessor(TimeoutError=-1, kernel_name='python3')
    execute_preprocessor.preprocess(notebook_content, {'metadata': {'path': '.'}})

