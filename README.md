This package contains Miroslaw's code base for Kaggle's see-click-fix competiton

This has been tested on ubuntu 12.04 LTS and should work on any standard
linux distribution

Installation
------------
    sudo apt-get install python-scipy ipython python-sklearn python-pandas

Running the winning model
-------------------------
    unzip -d data data/train.zip 
    unzip -d data data/test.zip
    unzip -d data data/geodata.zip
    unzip -d data data/bryans_data.zip
    mkdir cache, submissions

    python test.py ridge39

This will create all datasets required for the model, train the model, and save
the predictions. The final model and intermediate datasets will be saved to cache.

By default the predictions will be saved to submissions/ridge39.csv
In general predictions are saved using the format $SUBMITDIR/$MODELNAME.csv

To change the default location for predictions and cache modify the global variables
in config.py

A note on the winning model: 
It was recently discovered that when I started using Bryan's dataset to explore
his population-based features with my model there were some preprocessing steps
he took with the summary and description fields I did not notice. As a result,
to get exactly reproducable results for the winning model, you have to run 
the code with Bryan's data. There is a boolean flag inside of config.py that 
controls this. If you would like to try out my model without relying on Bryan's 
data you can set the boolean to False. 

Accessing the model and data in python
-------------------------------------- 
    import models
    model = models.train_model('ridge39')
    # I know the naming is poor at the moment, but after the model has been 
    # trained once it will be pulled from cache
    
    # The same goes for datasets associated with the model, they get pulled
    # from cache after being created once
    train_data, test_data = models.get_model_data('ridge39')

A brief summary of the code
---------------------------
All dataset and model configurations are set in settings.json

settings.json is a json object with two nested objects "models" and "datasets"

The format for defining a dataset is
    "DatasetName": { "input_data": ["Names", "Of", "Input", "Datasets"]
                     "transforms": [["transform_name1", {"args": "values"}],
                                    ["transform_name1", {"args": "values"}]] }

The format for defining a model is
    "ModelName": { "model": "PythonNameForModelObject", 
                   "dataset": "NameOfDatasetUsedByModel",
                   "target": "NameOFDatasetForTargetVariables",
                   "args": {"NameOfArgument": "ValueOfArgument", ...},
                   "validator": { "name": "NameOfCrossValidaton",
                                  "args": "ArgsForCrossValidation" }
                 }
                 
Then these datasets and models can be used within python as shown above
