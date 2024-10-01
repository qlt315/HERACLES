function [out_hpstream,out_lpstream] = f_genstream(in_file)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    
    dct_out = dct2(in_file);
    
    dct_quant = int32(dct_out);
    
    cutoff = 0.5*length(dct_quant);
    %%% low priority stream (high frequency)
    Low_T = fliplr(tril(fliplr(dct_quant), cutoff));
    
    %%% serialize Low_T in stream 
    out_lpstream = reshape(Low_T,1,numel(Low_T))';
    
    %%% high priority stream (Low frequency)
    High_T = dct_quant - Low_T;
    %%% serialize Low_T into stream
    out_hpstream = reshape(High_T,1,numel(High_T))';
   
end

