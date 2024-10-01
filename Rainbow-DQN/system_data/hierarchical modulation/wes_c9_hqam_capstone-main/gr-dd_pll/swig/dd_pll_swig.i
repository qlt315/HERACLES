/* -*- c++ -*- */

#define DD_PLL_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "dd_pll_swig_doc.i"

%{
#include "dd_pll/qam_pll.h"
%}

%include "dd_pll/qam_pll.h"
GR_SWIG_BLOCK_MAGIC2(dd_pll, qam_pll);
