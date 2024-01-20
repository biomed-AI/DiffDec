# Datasets

To train the model from scratch, you need to download the preprocessed file from [this link]() and save it into `data/single` or `data/multi`.

We use the CrossDocked dataset and reaction-based slicing method in LibINVENT to construct single and multi R-groups datasets. If you want to process the dataset from scratch, you can follow the steps:

1. Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: 
```bash
tar -xzvf crossdocked_pocket10.tar.gz
```
3. Split raw PL-complexes and convert sdf files into SMILES format:
```bash
python split_and_convert.py
```
4. Use the reaction-based slicing method in LibINVENT to slice the molecules into scaffolds and R-groups in [Lib-INVENT-dataset](https://github.com/MolecularAI/Lib-INVENT-dataset) and replace `example_configurations/supporting_files/filter_conditions.json` in Lib-INVENT-dataset with `filter_conditions.json` in this directory.
For single R-group dataset, set the value of parameter `max_cuts` in `example_configurations/reaction_based_slicing.json` to `1` while for multi R-groups dataset to `4`.
5. Process and prepare datasets:
```bash
cd single
python -W ignore process_and_prepare.py
```
```bash
cd multi
python -W ignore process_and_prepare.py
```