/* -*- c++ -*- */

#define CHECKEVM_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "checkevm_swig_doc.i"

%{
#include "checkevm/getevm.h"
%}

%include "checkevm/getevm.h"
GR_SWIG_BLOCK_MAGIC2(checkevm, getevm);
