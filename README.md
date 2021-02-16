# TMR4222 Heat Transfer Course Supplement Material

## Setting up the working environment
These materials are written in Jupyter Notebook, which is based on Python. You should have a 
minimum knowledge of Python to manipulate it. Don't worry. You will be still see and run the 
code as it is. But you still need to set up the environment to do that. 

### Install the right version of Python (>=3.8)
First, you need Python installed on your PC. The version should be at least 3.8 or up. The 
easiest way to check if you have the right version of python is to open a terminal or command 
prompt and enter the following command.
```commandline
python --version
```
You can also try
```commandline
python3 --version
```
Otherwise, you can easily install Python by going [here](https://www.python.org/downloads/). 
Don't change the options for the install if you don't know what you are doing. After 
installation, check the version as above to make sure that you have got the right python installed. 

### Install Git
This step is optional. You can download the repository from the webpage as a zip file. But, I 
strongly recommend you to learn about Git if you haven't used it yet. You can install Git from 
[here](https://git-scm.com/downloads). If you are not sure if you have git already installed or 
not, just try following command on your terminal:
```commandline
git --version
```
Once you installed the git, open the terminal and go to any folder that you like to place the 
material in the terminal and type the following:
```commandline
git clone https://github.com/kevinksyTRD/HeatTransferCourse.git
```

If you don't like a terminal, Github Desktop is an option. You can download it 
[here](https://desktop.github.com/).

### Create a virtual environment
In order to prevent any conflict with the packages that you already have installed and what will 
work for these notebooks, I recommend creating a virtual environment. First, you have to install 
the virtual environment package if you haven't. I will assume that you can enter 'python' in any 
directory to run Python. First, you need to be in the directory of these materials.
```commandline
cd HeatTransferCourse
pip install virtualenv
```
Then, create a virtual environment.
```commandline
python -m virtualenv venv
```
This will create your virtual environment in 'venv' directory. Next you can activate the virtual 
environment. If you are on Windows PC, 
```commandline 
venv\Script\activate
```
Otherwise,
```commandline
source venv/bin/activate
```
NB! Whenever you restarted the terminal, make sure that you activate the virtual environment using 
the above command.

### Install Packages
Once you have activated your virtual environment, you can now install your packages. You must be 
connected to internet to do so.
```commandline
pip install -r requirements.txt
```
This will install all the necessary packages to run the notebooks.

### Run Jupyter Lab
Launch Jupyter Lab with:
```commandline
jupyter-lab
```