Setup:

1. Download the database:

- with command: kaggle datasets download -d viratkothari/animal10
- by hand: https://www.kaggle.com/viratkothari/animal10 => download

and place it in folder /datasets/Animal-10

2. Prepare python (3.9.5+):

- python -m pip install -r requirements.txt

3. Prepare the dataset:

- python prepare_dataset.py

If everything worked without errors, you should be able to run:

- python main.py
