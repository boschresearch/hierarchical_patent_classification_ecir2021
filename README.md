
## A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers
This is the companion code for the experiments reported in the paper "A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers"  by 
Subhash Chandra Pujari, Annemarie Friedrich and Jannik Strötgen published at ECIR 2021.
The code allows the users to reproduce the results reported in the paper and extend the model to 
new datasets and hierarchical multilabel classification configurations. 
For any queries regarding code or dataset, you can contact [Subhash Pujari](subhashchandra.pujari@de.bosch.com). 
Please cite the paper when reporting, reproducing or extending the results as:
## Citation
```
@inproceedings{pujari-etal-2021,
    title = "A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers",
    author = {Subhash Chandra Pujari and Annemarie Friedrich and
      Str{\"o}tgen, Jannik},
    booktitle = "Proceedings of the 43rd EUROPEAN CONFERENCE ON INFORMATION RETRIEVAL",
    month = mar,
    year = "2021",
    address = "Online"
}
```

## Dataset
Dataset can be found in the folder as data/uspto-release.

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication 
"A Multi-Task Approach to Neural Multi-Label Hierarchical Patent Classification using Transformers", ECIR 2021. It will
neither be maintained nor monitored in any way.

## Setup
* pip install -r requirements.txt
* update the config file to provide directory paths and model parameters. Sample config can be found in ./config/ folder.
* python main.py --config-file <path-to-config-file> --op <train/test>

## License
The code is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

A data set sampled from USPTO coprus is located in the folder [data/uspto-release](data/uspto-release) are
licensed under a [Creative Commons Attribution 4.0 International License] (http://creativecommons.org/licenses/by/4.0/) (CC-BY-4.0).

 