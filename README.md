# UML-project
## Getting the data
Because there is a lot of data, this is how to get it and change it.

First make sure git lfs is installed (it helps store large files). If not
run the following commands.
```
wget https://github.com/git-lfs/git-lfs/releases/download/v2.6.0/git-lfs-linux-amd64-v2.6.0.tar.gz
tar -xf git-lfs-linux-amd64-v2.6.0.tar.gz
sudo ./install.sh
```

Then simply do a git pull, this (given that you installed git lfs correctly)
should pull al large file called data.zip into your local repo. Finally just
type
```
make getdata
```

This should create a folder called data/ with lots of data in it.

## Changing the data
If you make changes to the data simply use the command:
```
make zipdata
```
You can then `git add` and `git commit` as usual.
