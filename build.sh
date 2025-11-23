# download & install Miniconda silently
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda

# make conda commands available in this shell
export PATH="$HOME/miniconda/bin:$PATH"
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# create/update the env from your environment.yml
conda env create -f environment.yml -n myenv || conda env update -f environment.yml -n myenv

which conda

# activate and install any remaining pip deps if needed
conda activate myenv