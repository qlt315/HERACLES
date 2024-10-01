function [SourceCoded_HPSignal,dict_hp, SourceCoded_LPSignal, dict_lp] = f_getsourcecode(hp_stream,lp_stream)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

hp_quant = double(hp_stream);
lp_quant = double(lp_stream);

% find pdf of hp stream to be used in huffman encoding
sym_hp = unique(hp_quant(:));
count_hp = hist(hp_quant(:),sym_hp);
pdf_hp = double(count_hp)./sum(count_hp);

% find pdf of lp stream to be used in huffman encoding
sym_lp = unique(lp_quant(:));
count_lp = hist(lp_quant(:),sym_lp);
pdf_lp = double(count_lp)./sum(count_lp);

% generate huffman dictionary for both lp and hp data
dict_hp = huffmandict(sym_hp,pdf_hp); %  Dictionary creation i.e assigning codes to each data symbol
dict_lp = huffmandict(sym_lp,pdf_lp);

SourceCoded_HPSignal = huffmanenco(hp_quant,dict_hp); %  source coding for high prority scheme.
SourceCoded_LPSignal = huffmanenco(lp_quant,dict_lp); % source coding for low prority scheme.

end

