# cyamese

Siamese network for machine learning flow cytometry data for Python2.

First pip install virtualenv if you don't already have it.

```
pip2 install virtualenv
```

Create and activate your environment. Note you may want to include
--system-site-packages to inherit global packages (on GPU, include this).

```
virtualenv --python=/usr/bin/python2.7 cypy #--system-site-packages
source cypy/bin/activate
```

If you're running on a local computer, use:

```
pip install -r config_files/requirements_local.txt
```

If you're running on the GPU, use:

```
pip install -r config_files/requirements_gpu.txt
```

Once downloaded, you should have the fcm library loaded. There is a 
bug in the fcm library script gate.py. I've provided the working version
here. Replace it:

```
cp config_files/gate.py cypy/lib/python2.7/site-packages/fcm/core/gate.py
```

Then cd into the cyamese_scripts directory and run the cyamese script. First argument
should be the path/to/directory containing the fcs files and the metadata pickle file
as created by apidownload.py (instructions for this coming soon!).

```
python cyamese.py allimmport/
```


