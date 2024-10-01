/* -*- c++ -*- */

#define DCT_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "dct_swig_doc.i"

%{
#include "dct/dct_ff.h"
%}

%include "dct/dct_ff.h"
GR_SWIG_BLOCK_MAGIC2(dct, dct_ff);
