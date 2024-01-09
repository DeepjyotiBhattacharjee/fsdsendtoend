echo [$(date)]: "START"

echo [$(date)]: "creating env with python 3.11.3 version"

conda create --prefix ./env python=3.11.3 -y

echo [$(date)]: "Activating the environment"

source activate ./env 

echo [$(date)]: "Installing the dev requirements"

pip install -r requirements.txt

echo [$(date)]: "END" 

