This is a draft readme for how to train/retrain the am2 model. It will be refined in a later branch/commit.

To check for newly available data for processing: python -m projects.am2_project.src.check_for_data from the root dir of the repo

This checks in ./projects/am2_project/models/all_item_models for previously trained models.

It checks in ./projects/am2_project/config/am2_config.json for project details (sweeps, and items from sweeps)

It checks for new data for the project, and if it finds any, creates input features for those and
saves them to ./projects/am2_project/data/pending_training_data

*****

Afterwards, when :

you read data in from ./projects/am2_project/data/pending_training_data/am2_relationships_data_for_future_model

you read models in from ./projects/am2_project/models/all_item_models

then you run train_semi_supervised_model. when it creates the model for a particular item
type, it will save it in models/(item_type), and also in models/all_item_models