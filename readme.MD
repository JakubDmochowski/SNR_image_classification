Setup:

1. Download the database:

- with command: kaggle datasets download -d viratkothari/animal10
- by hand: https://www.kaggle.com/viratkothari/animal10 => download

unzip it and place it in folder /datasets/Animal-10

2. Prepare python (3.9.5+):

- python -m pip install -r requirements.txt

3. Prepare the dataset:

- python prepare_dataset.py

If everything worked without errors, you should be able to run:

- python main.py

### Deepdream
1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` from project directory (this will create a brand new conda environment).
3. Run `activate pytorch-deepdream` (for running scripts from your console or setup the interpreter in your IDE)
4. Run deepdream with chosen layer and input image eg.
    `python deepdream/deepdream.py --input dog(64).jpeg --model_name RESNET18 --layers_to_use layer30`