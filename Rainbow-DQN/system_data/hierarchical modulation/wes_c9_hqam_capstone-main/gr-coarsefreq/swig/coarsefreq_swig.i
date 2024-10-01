/* -*- c++ -*- */

#define COARSEFREQ_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "coarsefreq_swig_doc.i"

%{
#include "coarsefreq/c_freq_offset.h"
%}

%include "coarsefreq/c_freq_offset.h"
GR_SWIG_BLOCK_MAGIC2(coarsefreq, c_freq_offset);
