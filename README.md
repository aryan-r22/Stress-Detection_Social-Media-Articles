# Stress Detection from Social Media Articles: New Dataset Benchmark and Analytical Study
This repository contains the datasets for **classification of stress** from text-based social media articles from Reddit and Twitter, which were created within the paper titled "Stress Detection from Social Media Articles: New Dataset Benchmark and Analytical Study". 

***Presentated orally at IEEE WCCI 2022, IJCNN track (DOI = [10.1109/IJCNN55064.2022.9892889](https://doi.org/10.1109/IJCNN55064.2022.9892889))***

## Overview of the datasets
We construct four high quality datasets using the text articles from Reddit and Twitter. Against each of the articles is a class label with a value of '0' or '1', where '0' specifies a *Stress Negative* article and '1' specifies a *Stress Positive* article. Annotation was done using an automated DNN-based strategy highlighted in the aforementioned study.

The description about each of the datasets is given as under:

- **Reddit Title**: Consists of titles from the articles collected from both stress and non-stress related subreddits from Reddit.  
- **Reddit Combi**: Consists of title and body text combined together to form a single text sequence, collected from both stress and non-stress related subreddits from Reddit.  
- **Twitter Full**: Consists of stress and non-stress related tweets, collected from Twitter.  
- **Twitter Non-Advert**: Consists of the denoised version of the **Twitter Full** dataset.  

The details about the datasets may be directly referred to from the study.

## Other Files
Code files for data preprocessing and finetuning DistilBERT on stress detection are also provided. The files are: 

- **distilbert_train.py**: Finetuning script for DistilBERT.
- **distilbert_eval.py**: Evaluation script for DistilBERT.
- **reddit_preprocessing.py**: Preprocessing script for Reddit dataset. 
- **twitter_preprocessing.py**: Preprocessing script for Twitter dataset.   

The automated annotation code may be provided on request to the authors.

## Citation
```
@inproceedings{rastogi2022stress,
  title={Stress Detection from Social Media Articles: New Dataset Benchmark and Analytical Study},
  author={Rastogi, Aryan and Liu, Qian and Cambria, Erik},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```
