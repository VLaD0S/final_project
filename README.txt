INSTRUCTIONS:

ENVIRONMENT:
All extermal python packages are inside the requirements.txt file.
Create a python 3.5 virtual environment.
Linux install instructions:

	1. sudo apt-get install python3-pip
	2. sudo pip3 install virtualenv
	3. (in the source folder)  virtualenv env
	4. source env/bin/activate
	5. (once in the virtual environment) pip install -r requirements.txt

In order for everything to function, ensure that the following project structure is present in the root folder:

Python files:

- cnn_algoritm.py
- data_loader.py
- load_model.py
- graph_freezer.py

Directories:

- data
- models
- <name of folder containing data to be trained, in this case, Skin_Data_All>
 
- The structure of the folder containing the data to be trained must be of the strucutre:

>dataset name (dir)
	>label 1 (dir)
			fil21.jpg
			file2.jpg
			...
	>label 2 (dir)	
			file.jpb
			...
	...





--Training and Model INFO--
Data used to train these models:
Skin_Data.tar.gz

Links for trained models:

For retrained inception model:
https://drive.google.com/open?id=1I8W8McnIX_LRaCagb0USj9Mzm9sRMwdK
