# credit score prediction service
#### Description: service to predict client credit score

## Requirements
#### `python >= 3.10` + Pipfile dependencies

## Install

#### 1. Create `.venv` folder at root path of the project
#### 2. Run `pipenv shell` to activate virtual environment
#### 3. Run 
- `pipenv install` to install the latest versions of dependencies
- `pipenv install --ignore-pipfile` to install used versions of dependencies at the moment of development
  
## Run:
```
python3 -m hypercorn main.py or python3 -m uvicorn main:app
```
