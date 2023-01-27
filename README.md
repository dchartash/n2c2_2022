# Yale Submission to n2c2 2022: Track 3 Challenge

This is the code used in training Yale's submission to the [n2c2 2022 Challenge: Progress Note Understanding: Assessment and Plan Reasoning  ](https://n2c2.dbmi.hms.harvard.edu/2022-track-3), achieving  **2nd place** model among 14 teams. The model was trained with huggingface transformers and PyTorch, using Yale High Performance Computing Cluster (YCRC) GPU resources.

Configuration files are all located under the [conf/](https://github.com/dchartash/n2c2_2022/tree/main/conf) directory. 

Further investigation was done to explain model predictions using SHAPley values and the output html files are under [explainability/](https://github.com/dchartash/n2c2_2022/tree/main/explainability) with the code to perform the analysis under [Shap_Viz.ipynb](https://github.com/dchartash/n2c2_2022/blob/main/Shap_Viz.ipynb). Due to their size and interactive nature, these will need to be downloaded and opened locally. 

# Training and Inference

All relevant final training code can be found [trainer_with_inference.py](https://github.com/dchartash/n2c2_2022/blob/main/trainer_with_inference.py) and [gpt_inference.py](https://github.com/dchartash/n2c2_2022/blob/main/gpt_inference.py) for the GPT-2 model comparison. 

# Citation

The publication is currently _Under Review_ at Journal of Biomedical Informatics (JBI) and the bibtex will be posted as soon as the paper is accepted. 
