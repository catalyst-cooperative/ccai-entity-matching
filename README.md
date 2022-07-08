# ccai-entity-matching
This repo contains preliminary results from entity matching FERC1 and EIA data with Panda for the CCAI grant.

The `notebooks` directory contains downloaded notebooks and logs from Panda. The notebooks are labeled with numbers that correspond to the CSV of exported matched records generated from this notebook. The CSV's of matched records generated from these notebooks are contained in the `panda_matches` directory.

I've added the input zip file I've been using for Panda in the `panda_inputs` directory of this repo. This zip file contains a left and right CSV where `left.csv` is the 2020 FERC 1 records and `right.csv` is the distinct (only true granularities) 2020 EIA plant parts records. Some cleaning was done and also utility names are merged onto these datasets. These inputs probably shouldn't be checked-in in the future, and will be replaced by a preprocessing function to generate the left and right data frames.

`train_ferc1_eia.csv` is the training data labels of matched EIA and FERC records, taken from the RMI repo. The 2020 records are separated out from this in `train_ferc1_eia_2020.csv`. I need to check with people working on RMI stuff to make sure this is the most up to date training data.

The notebook `training_data_metrics` contains code to compare the Panda found matches with the hand labeled training data matches.

So far Panda is only finding 24 of the 115 labeled matches for 2020. I take a deeper look at this in the notebook, but I think this is most likely because of NaN values.
