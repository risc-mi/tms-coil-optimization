# Real-Time Estimation of the Optimal Coil Placement in Transcranial Magnetic Stimulation Using Multi-Task Deep Learning

## Description

* This repository contains the source code for the multi-task deep neural network described in (Moser et al., 2024, doi: tbd).
* All imaging data used in this study is open and publicly available at the respective project webpage: the Human Connectome Project (\url{https://www.humanconnectome.org/study/hcp-young-adult}) and the University of Pennsylvania glioblastoma data (\url{https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642}).


## Running a training and inference

* Start training and inference via:
```
python train_predict.py
```
* the results will be stored in the "predictions" folder


## Sample Data
* In the sample_data folder, we provide an exemplary sample from the HCP database
* affine_matrix: contains the affine matrix describing the optimal coil placement (obtained from SimNIBS's TMSoptimize function)
* t1: subject T1 MR images
* tissues: subject tissue segmentation (obtained from SimNIBS directly)
* targetdistance_efield: contains both the cortical stimulation target (transformed into a distance field) and the induced electric field (model output)


## Citation
Moser, P.; Reishofer, G.; Prückl, R.; Schaffelhofer, S.; Thumfart, S.; Mahdy Ali, K.; "Real-Time Estimation of the Optimal Coil Placement in Transcranial Magnetic Stimulation Using Multi-Task Deep Learning", *Scientific Reports*, **2024**, pp. TBD, doi: TBD  

## Acknowledgements
This project is financed by research subsidies granted by the government of Upper Austria (research project MIMAS.ai) and by the FFG (research project nARvibrain, grant no. 894756). www.ffg.at. “ICT of the Future” programme – an initiative of the Federal Ministry for Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK). RISC Software GmbH is a member of UAR (Upper Austrian Research) Innovation Network.
