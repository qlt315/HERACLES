/* -*- c++ -*- */

#define FLL_EST_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "fll_est_swig_doc.i"

%{
#include "fll_est/my_fll.h"
%}

%include "fll_est/my_fll.h"
GR_SWIG_BLOCK_MAGIC2(fll_est, my_fll);
