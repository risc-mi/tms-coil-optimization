# Real-Time Estimation of the Optimal Coil Placement in Transcranial Magnetic Stimulation Using Multi-Task Deep Learning

## Description

* This repository contains the source code for the multi-task deep neural network described in (Moser et al., 2024, doi: tbd).
* All imaging data used in this study is open and publicly available at the respective project webpage: the Human Connectome Project (\url{https://www.humanconnectome.org/study/hcp-young-adult}) and the University of Pennsylvania glioblastoma data (\url{https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642}).

## Step-by-step guideline for generating training/test data
* Download subject T1+T2 MR images (for example from HCP database)
* Run SimNIBS's charm pipeline on each subject using T1+T2 MRI
* Select cortical stimulation targets on gray matter mesh (manual or automated), i.e., get the coordinates of these points
* Run SimNIBS's "TMSoptimize" function on all these targets (run also "mesh-to-nifti" function afterwards)
* The resulting affine matrix from SimNIBS contains the optimal coil position and optimal rotation matrix
* The cortical target coordinates as transformed to a distance field
* We resampled all 3D volumes (T1, tissue segmentation, target-distancefield and induced eletric field) to 112x96x80
* For more information, please refer to the original article (Moser et al., 2024, doi: tbd)


## Sample Data
* In the sample_data folder, we provide some exemplary samples from the HCP database
* affine_matrix: contains the affine matrices describing the optimal coil placements
* target_coords: contains the coordinates of the cortical stimulation targets
* t1: subject T1 MR images
* tissues: subject tissue segmentations
* targetdistance_efield: contains both the cortical stimulation target as distance field and the induced electric field

* Model input: T1 + tissue segmentation + target-distancefield
* Model output 1: induced electric field
* Model output 2: coil position
* Model output 3: coil orientation (Euler angles calculated from optimal rotation matrix)

## Running a training and inference
* Start training and inference via:
```
python train_predict.py
```
* the results will be stored in the "predictions" folder

## Citation
Moser, P.; Reishofer, G.; Prückl, R.; Schaffelhofer, S.; Thumfart, S.; Mahdy Ali, K.; "Real-Time Estimation of the Optimal Coil Placement in Transcranial Magnetic Stimulation Using Multi-Task Deep Learning", *Scientific Reports*, **2024**, pp. TBD, doi: TBD  

## Acknowledgements
This project is financed by research subsidies granted by the government of Upper Austria (research project MIMAS.ai) and by the FFG (research project nARvibrain, grant no. 894756). www.ffg.at. “ICT of the Future” programme – an initiative of the Federal Ministry for Climate Action, Environment, Energy, Mobility, Innovation and Technology (BMK). RISC Software GmbH is a member of UAR (Upper Austrian Research) Innovation Network.
