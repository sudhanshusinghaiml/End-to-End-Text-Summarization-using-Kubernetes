import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_sorce_code_directory = 'text_summarization'

list_of_files = [
    f'src/__init__.py',
    f'src/{project_sorce_code_directory}/__init__.py',
    f'src/{project_sorce_code_directory}/components/__init__.py',
    f'src/{project_sorce_code_directory}/components/data_ingestion.py',
    f'src/{project_sorce_code_directory}/components/data_validation.py',
    f'src/{project_sorce_code_directory}/components/data_transformations.py',
    f'src/{project_sorce_code_directory}/components/model_trainer.py',
    f'src/{project_sorce_code_directory}/components/model_evaluation.py',
    f'src/{project_sorce_code_directory}/config/__init__.py',
    f'src/{project_sorce_code_directory}/config/configuration.py',
    f'src/{project_sorce_code_directory}/entity/__init__.py',
    f'src/{project_sorce_code_directory}/constants/__init__.py',
    f'src/{project_sorce_code_directory}/exception/__init__.py',    
    f'src/{project_sorce_code_directory}/logger/__init__.py',
    f'src/{project_sorce_code_directory}/pipeline/__init__.py',
    f'src/{project_sorce_code_directory}/pipeline/data_ingestion_pipeline.py',
    f'src/{project_sorce_code_directory}/pipeline/data_validation_pipeline.py',
    f'src/{project_sorce_code_directory}/pipeline/data_transformation_pipeline.py',
    f'src/{project_sorce_code_directory}/pipeline/model_training_pipeline.py',
    f'src/{project_sorce_code_directory}/pipeline/model_evaluation_pipeline.py',
    f'src/{project_sorce_code_directory}/pipeline/prediction_pipeline.py',
    f'src/{project_sorce_code_directory}/utils/__init__.py',
    f'src/{project_sorce_code_directory}/utils/common_utils.py',
    'app.py',
    'requirements.txt',
    'Dockerfile',
    'setup.py',
    'params.yaml',
    'config/config.yaml'
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")