# AI Scholar Dataset

The dataset is constructed for deeper analysis of scholars as well as papers in the AI community.

We provide public access to the two collections below:
- Download the 78K Google Scholars data through the Google Drive shared link: [gs_scholars.npy](https://drive.google.com/file/d/1sfNLH549c0IMp-hojnpmskBftsW5jB7a/view?usp=sharing).
- Download the 2.6M papers data through the Google Drive shared link: [ai_paper_features_100k](https://drive.google.com/file/d/16cmOlJ-8--7vqIXY-hP0JXtRwqaPoOfh/view?usp=sharing).
## Collection of 78K Google Scholars

The data contains 78,536 AI scholars with all features directed obtained from Google Scholar profile pages. We crawled the list of AI Scholars through four domain tags shown on the Google Scholar profile page: AI, MLP, ML, CV. To control the scale of the dataset, we includes scholars with total citations over 100 by Jan 1, 2022.

[AIScholars78k_samp1000.csv](data/AIScholars78k_samp1000.csv) shows 1000 random samples of the dataset.


## Collection of 2.8M Papers

The data contains 2,890,908 AI papers. We collected all paper titles by iterating through the Google Scholar profile of each AI researcher by Jan 1, 2022.

[Papers100k_samp1000.csv](data/Papers100k_samp1000.csv) shows 1000 random samples of the data.