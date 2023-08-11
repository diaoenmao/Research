# Cluster Access and Use

## Prerequisites

Contact enmao.diao@duke.edu, yuting.ng@duke.edu and vahid.tarokh@duke.edu to be added to the relevant
group for access to the cluster machines

## Machines

:warning:
Our servers are not for CPU computing. CPU intensive programs may jam
other users' GPU processes and may shut down the server due to the CPU RAM
limit.
If you want to access high performance computing for CPU, you should check the
Duke Computing Cluster which provides abundant CPU computing resources.
https://dcc.duke.edu/

The GPU Cluster lives at research-tarokhlab-01.oit.duke.edu to
research-tarokhlab-14.oit.duke.edu

|   Machine   |   GPU    |
| ----------- | -------- |
| 1           | Titan XP |
| 2-3         | RTX 2080 |
| 4-14        | RTX 3090 |

Each server node has 4 GPUs.

### File Storage

Each machine has some local and some remote storage attached

/scratch – very fast, local NVME drives on each server, ~7TB each.  No
redundancy and no backups, data loss can occur due to failures, system
activities, and manual errors.  Use: storage needed during computation and model
development.

/hpc/home – standard speed network attached storage, a single volume that will
be mounted across all servers.  Robust enterprise storage with internal
redundancy and full backups have been enabled.  Initial size 1TB. Use: home
directories for individual lab members.

/hpc/group/tarokhlab - standard speed network attached storage, a single volume
that will be mounted across all servers.  Robust enterprise storage with
internal redundancy, (no backups but there is a self-service 7-day snapshot).
This is the 1TB existing volume that we mentioned that Research Computing
provides at no cost as a resource as a general-purpose storage volume across
some RC services.  It is also mounted to the Duke compute cluster and available
via Globus for transferring files in and out of RC storage. Use: shared data
sets for the lab, or other project-based resources that will be used across the
lab.

## SSH Access and setup.

Generate SSH Keys

```
ssh-keygen -t ed25519 -C "your_email@duke.edu"
```

Follow the prompt and save your SSH key with some descriptive name like id_duke.
It is highly recommended to use a strong passphrase for your ssh key, especially if
this key will be used in some shared environment. Next, view your ssh **public** key using


```
cat ~/ssh/id_duke.pub
```

Copy it and paste it to https://idms-web-selfservice.oit.duke.edu/ under "Advanced IT
Options"

### Adding key to your SSH Config

On Mac and Linux, you can add the SSH key to `$HOME/.ssh/config`, such that
it'll automatically use the correct key for these servers. A side benefit is
that both systems can be set up to tab complete the server name by typing `ssh`
at the terminal. A sample config entry is as follows:


```
Host research-tarokhlab-04 research-tarokhlab-05 research-tarokhlab-07 research-tarokhlab-11
     HostName %h.oit.duke.edu
     User     gl161
     PreferredAuthentications publickey
     IdentityFile       ~/.ssh/id_duke

```

This sets up all hosts listed on the first line, you can enumerate through all
the hosts we have available, or just set the ones you use frequently. On Ubuntu,
this allows me to quickly tab complete for my ssh command, but on Mac there is a
bit of setup for zsh tabcomplete.

Finally, to access the machine, simply type `ssh research-taroklab-05` for
example.

To get files to and from a machine

```
scp [your file] research-tarokhlab-XX

scp research-tarokhlab-XX:/path/to/file [local path to store file]
```

## Working with the GPUs

To check on the gpu utilization and health, run

```
nvidia-smi
```

To train with a specific subset of GPUs, use

```
CUDA_VISIBLE_DEVICES="0" python train_model.py
```

To kill a process list by NVIDIA-SMI

```
kill -9 [process d]
```

To kill all your processes on all GPUs

```
kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')
```

## First time setup with Anaconda and PyTorch.

:warning: You should not save your model in your home directory /hpc/home.

The SSD device is mounted at /scratch and /hpc/group/tarokhlab.
See [File Storage](#file-storage) for details

### Installing Anaconda in /scratch (Recommended)
Note that since /scratch is per machine, you'll have to repeat these steps for
different machines.

```
mkdir /scratch/{netid}

cd /tmp

curl https://repo.anaconda.com/archive/Anaconda3-2023.07-0-Linux-x86_64.sh
--output anaconda.sh

bash anaconda.sh
```

Press `q` to jump to the end of the license agreement (or read through it if you
really want to), and type `yes` to continue. When the installer asks for the
location, type /scratch/{netid}/anaconda3. At the end, type yes for it to
automatically set up Anaconda for you.

Lastly, exit the ssh connection with Ctrl-D and re-ssh back to the machine for
your new changes to take effect.

To verify that it's installed correctly, type

```
which conda
```

and that should show /scratch/{netid}/anaconda3/bin/conda

### Setting up Anaconda

Note, from this point on, I usually do everything inside of a tmux (or screen)
session. See [Terminal Multiplexers](tmux.md)

While you can use just the base environment in Anaconda, it is best practice to
create a new environment for each project. This avoids dependency issues, for
example where project A only works with CUDA 11.7 and project B only works with
CUDA 11.8

To create a new project:
```
conda create -n project_name
conda activate project_name
```

To install packages (e.g. PyTorch and jupyter-notebook):
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge jupyter_contrib_nbextensions   (Optional, may take very long)
```
Note that the -c flag specifies a channel to install from. By default you do not
need it.

### Setting up Jupyter Notebook and connecting to it

Choose a port number, like 1234, then

```
jupyter notebook --no-browser --port=1234
```
Copy the `http://localhost:1234/?token=....` line and save it, you will need it
to connect.

Next, detach from your tmux or screen session, close the ssh connection and
reconnct using:

```
ssh -L 1234:localhost:1234 research-tarokhlab-XX
```
This binds port 1234 on your local machine to research-tarokhlab-XX's localhost:1234.
Note that these two numbers can be different if you really want them to.

Paste the link you saved earlier into the address bar of your web browser on
your machine. Note that if you specified a different local port from the one you
told jupyter-notebook to use, you'll need to change the port in the URL.
