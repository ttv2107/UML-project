# Denoising Face Recognition via DBSCAN
## Authors
Trung Vu
Sihyun Lee
Alexandre Lamy

## Introduction
This is the Github repo for the Denoising Face Recognition via DBSCAN project that we worked on for 
the class COMS 4995: Methods in Unsupervised Learning (Fall 2018). It contains the code and data
that we used as well as the results we obtained. It also has our final poster and writeup.

## Viewing poster, report, and results
The poster is the file `final_poster.pdf`, the final report is the file `final_report/final_report.pdf`. Results can all be found in the `results/` folder.

## Getting data and running the code
### Cloning this repo
To clone this repo first make sure that 
[git lfs](https://git-lfs.github.com/) is 
installed (it helps store large files). If not
run the following commands.
```
wget https://github.com/git-lfs/git-lfs/releases/download/v2.6.0/git-lfs-linux-amd64-v2.6.0.tar.gz
tar -xf git-lfs-linux-amd64-v2.6.0.tar.gz
sudo ./install.sh
```

Once this is done you can `git clone`, `git pull`, `git add`, and `git commit` as usual.

### Getting the data
Our raw data (the images on which the neural net was trained, the neural net parameters, etc.)
are all in the `data/` folder. However, due to the very large number of files in this folder we do NOT
store it directly on Github. Instead, we zip it and store the `data.zip` file using `git lfs`. To 
retrieve the `data/` folder, first make sure your repo is up to date (do a `git pull`) and then simply
run `make getdata`, this will automatically unzip `data.zip` and create the `data/` folder.

### Code requirements
For the code to work, please make sure you are running `python 3.6` and install the python packages 
`numpy`, `scikit-learn`, and `opencv-python`.

### Running the code
Almost all our code is broken down into independent and self contained scripts that take in command line arguments. Each can be called with the `-h`
flag to see details.

The parameters for the Neural Network which creates the image embedding is found in file `data/vgg2.pb`. These weights come from the `vgg2` dataset (we did not retrain).
We used standard `facenet` scripts to train an SVM classifier over the 5 faces we were interested in detecting. These scripts are not included but are easy to find online (see [facenet](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images)).
The resulting SVM model is included in `code/classifier.pkl`. The file `code/classify_video.py` is a simple script that uses that classifier to detect and classify faces in a video.

The code for our post processing method can then be found in the file `code/dbscan_denoising_alg.py`. 

### Changing the data and committing
If you make changes to the data (try new videos or people, change model weight, etc.) there is a simple procedure to push those changes to Github.

Simply use the following command to automatically correspondingly update the `data.zip` file.
```
make zipdata
```
You can then `git add`, `git commit`, and `git push` as usual.
