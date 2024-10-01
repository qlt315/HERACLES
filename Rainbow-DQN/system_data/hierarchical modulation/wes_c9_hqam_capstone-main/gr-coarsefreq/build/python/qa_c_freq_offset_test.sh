#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/home/bhagat/gr-coarsefreq/python"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/bhagat/gr-coarsefreq/build/python":$PATH
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/bhagat/gr-coarsefreq/build/swig:$PYTHONPATH
/usr/bin/python3 /home/bhagat/gr-coarsefreq/python/qa_c_freq_offset.py 
