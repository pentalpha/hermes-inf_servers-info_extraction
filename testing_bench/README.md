# Testing Environment for Information Extraction from Emergency Call Transcriptions

## Run the benchmarks

The following sequence of commands will install the required packages, load a specialized dataset from HF and make inferences with several models.

```sh
pip install -r requirements.txt
python dataset_loading.py
python testing.py results/my_results.json
```