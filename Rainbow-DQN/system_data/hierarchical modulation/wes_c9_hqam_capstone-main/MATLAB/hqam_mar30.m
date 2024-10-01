clc;clear;close all;

% ******* Implementation of Hierarchichal Modulation using two QPSK Schemes as HP and LP and making one 16 QAM*********  

% Source coding 

% symbols = [1 2 3 4 5]; % Data symbols 
% p = [0.5 0.22 0.2 0.05 0.03]; % Probability of each data symbol

% *********** experiments start

file_color = imread('lenna_fourth.png');
    
file_gray = rgb2gray(file_color);

[hpstream, lpstream] = f_genstream(file_gray);

[SourceCoded_HPSignal, dict_hp, SourceCoded_LPSignal, dict_lp] = ...
    f_getsourcecode(hpstream,lpstream);

% Channel coding
trellis = poly2trellis([3],[4 5 7]);
% trellis = poly2trellis([4],[5 7]);
ChannelCoded_HPSignal = convenc(SourceCoded_HPSignal,trellis);  % channel coding for high priority scheme.
ChannelCoded_LPSignal = convenc(SourceCoded_LPSignal,trellis);  % channel coding for low priority scheme.

% plot()

[m,n] = size(ChannelCoded_HPSignal);
[r,c] = size(ChannelCoded_LPSignal);

% Making randomly generated HP and Low priority matrices equal
if m*n>r*c          % If bits in HP stream signal are more then the LP stream signal then truncate the HP stream bits that are over and above the LP stream
    ChannelCoded_HPSignal(r+1:end) = [];
%     ChannelCoded_LPSignal = [ChannelCoded_LPSignal; zeros(m-r,1)];
    TSid=0;                                 % Truncated stream Id
elseif m*n<r*c   % If bits in LP stream signal are more then the HP stream signal then truncate the LP stream bits that are over and above the HP stream
    ChannelCoded_LPSignal(m+1:end) = []; 
%     ChannelCoded_HPSignal = [ChannelCoded_HPSignal; zeros(r-m,1)];
    TSid=1;
end

% QPSK Modulation
[a,b] = size(ChannelCoded_HPSignal);
d=a*b;                                  % Total no of bits in the signal
e=d/2;
if rem(e,1)==0  % check if no of bits are even or odd
    flag=0;
else
    ChannelCoded_HPSignal= vertcat(ChannelCoded_HPSignal,1); %concatenate another bit with value '1' if no of bits are odd to complete the last symbol of QPSK
    ChannelCoded_LPSignal= vertcat(ChannelCoded_LPSignal,1);
    e=(d+1)/2;
    flag=1;
end
HP_Symbol = reshape(ChannelCoded_HPSignal,2,e); %QPSK symbol formation and putting them in a matrix column wise
LP_Symbol = reshape(ChannelCoded_LPSignal,2,e); %QPSK symbol formation and putting them in a matrix column wise

[f,g] = size(HP_Symbol);
A= []; B=[];
Phase_HP=[];Phase_LP=[];
for i=1:g
     A=QPSK_BitToPhase(HP_Symbol(1,i),HP_Symbol(2,i)); % Assignning each High priority symbol a phase
     Phase_HP=[Phase_HP,A];                                    % saving all phases in a matix
end
for i=1:g
     B=QPSK_BitToPhase(LP_Symbol(1,i),LP_Symbol(2,i)); % Assignning each Low priority symbol a phase
     Phase_LP=[Phase_LP,B];                                    % saving all phases in a matix
end

HP_BeforeScaling = exp(1j*Phase_HP); % QPSK high priotiy stream before scaling
LP_BeforeScaling = exp(1j*Phase_LP); % QPSK Low priotiy stream before scaling


% assigning high priotiy  [for HP point at 3+3i at pi/4 amplitude=sqrt(3^2+3^2)=sqrt(18)=4.24]
HP_Mod_Signal=4.24*exp(1j*Phase_HP); % assigning high priotiy 
% assigning low priotiy [for 1+1i i.e for LP point to be 1 unit distance around the HP point amplitude= sqrt (1^2+1^2)=sqrt(2)=1.41]
% LP_Mod_Signal=1.41*exp(1j*Phase_LP); 

LP_Mod_Signal=1.41*exp(1j*Phase_LP); 

subplot(2,2,1);plot(real(HP_BeforeScaling),imag(HP_BeforeScaling),'.'); title('QPSK Signal Before Scaling');
subplot(2,2,2);plot(real(LP_BeforeScaling),imag(LP_BeforeScaling),'.'); title('QPSK Signal Before Scaling');
subplot(2,2,3);plot(real(HP_Mod_Signal),imag(HP_Mod_Signal),'.');   title('After Scaling:HP Stream');
subplot(2,2,4);plot(real(LP_Mod_Signal),imag(LP_Mod_Signal),'.');  axis([-2 2 -2 2]);title('After Scaling:LP Stream');


C=[];
CombinedSignal=[];
for i=1:g
    C=HP_Mod_Signal(1,i)+LP_Mod_Signal(1,i);
    CombinedSignal=[CombinedSignal,C];
end
 
scatterplot(CombinedSignal); title('4/16 QAM Signal Before Transmission');  % scatter plot of received signal


SNR=[1:1:20];
% SNR=[20];
BER_HP=[];BER_LP=[];
[w,x]=size(SNR);
for index=1:x
    
%     index = 30;
    % Hierarchical Modulation i.e Transmission through AWGN Channel
    Hierarchical_Mod_RxSignal = awgn(CombinedSignal,index,'measured');

    % scatter plot of received signal
    scatterplot(Hierarchical_Mod_RxSignal);                              
    title('4/16 QAM Hierarchical Modulated Signal After Transmission');
    close;

    if index == 20 || index == 10
        % scatter plot of received signal at SNR '20'
        scatterplot(Hierarchical_Mod_RxSignal);                              
        title('4/16 QAM Hierarchical Modulated Signal After Transmission');  
    end

    % Demodulation
    [k,l] = size(Hierarchical_Mod_RxSignal);
    Firstbit_HP=[]; Secondbit_HP=[]; Firstbit_LP=[]; Secondbit_LP=[];
    Firstbit_HPsymbol=[]; Secondbit_HPsymbol=[]; 
    Firstbit_LPsymbol=[]; Secondbit_LPsymbol=[];

    for i = 1:l
        % Computing Real and Imaginary parts
        R = real(Hierarchical_Mod_RxSignal(1,i));  
        I = imag (Hierarchical_Mod_RxSignal(1,i));

        % Decision for HP stream: in QPSK if real&imaginary parts > 0 then 
        % it means the received symbol falls in First Quadrant and Txtd 
        % bits were 00
        if R >= 0 && I >= 0                     % 1st Quadrant
            Firstbit_HP = 0;
            Secondbit_HP = 0;
        elseif  R < 0 && I >= 0                % 2nd Quadrant
            Firstbit_HP = 0;
            Secondbit_HP = 1;
        elseif  R < 0 && I < 0                 % 3rd Quadrant
            Firstbit_HP = 1;
            Secondbit_HP = 1;
        elseif  R >= 0 && I < 0                % 4th Quadrant
            Firstbit_HP = 1;
            Secondbit_HP = 0;
        end


        % Decision for LP stream:
        if  (R >= 3) && (I >= 3)                 % 1st Quadrant around HP point (3,3) at origin i.e 3+3i
            Firstbit_LP = 0;
            Secondbit_LP = 0;
        elseif  (R>=0)&&(R<3)&&(I>=3)        % 2nd Quadrant around HP point (3,3) at origin
            Firstbit_LP = 0;
            Secondbit_LP = 1;
        elseif  (R >= 0)&&(R<3)&&(I>=0)&&(I<3)     % 3rd Quadrant around HP point (3,3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 1;
        elseif  (R >= 3) && (I >= 0) && (I < 3)          % 4th Quadrant around HP point (3,3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 0;
        elseif  (R>=-3)&&(R<0)&&(I>=3)        % 1st Quadrant around HP point (-3,3) at origin i.e -3+3i
            Firstbit_LP = 0;
            Secondbit_LP = 0;
        elseif  (R<-3)&&(I>=3)             % 2nd Quadrant around HP point (-3,3) at origin
            Firstbit_LP = 0;
            Secondbit_LP = 1;
        elseif  (R<-3)&&(I>=0)&&(I<3)          % 3rd Quadrant around HP point (-3,3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 1;
        elseif (R>-3)&&(R<0)&&(I>=0)&&(I<3) % 4th Quadrant around HP point (-3,3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 0;
        elseif  (R>=-3)&&(R<0)&&(I>=-3)&&(I<0) % 1st Quadrant around HP point (-3,-3) at origin i.e -3-3i
            Firstbit_LP = 0;
            Secondbit_LP = 0;
        elseif  (R<-3)&&(I>=-3)&&(I<0)       % 2nd Quadrant around HP point (-3,-3) at origin
            Firstbit_LP = 0;
            Secondbit_LP = 1;
        elseif (R<-3)&&(I<-3)              % 3rd Quadrant around HP point (-3,-3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 1;
        elseif  (R>=-3)&&(R<0)&&(I<-3)       % 4th Quadrant around HP point (-3,-3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 0;
        elseif  (R>=3)&&(I>-3)&&(I<0)         % 1st Quadrant around HP point (3,-3) at origin i.e 3-3i
            Firstbit_LP = 0;
            Secondbit_LP = 0;
        elseif  (R<3)&&(R>=0)&&(I>=-3)&&(I<0)   % 2nd Quadrant around HP point (3,-3) at origin
            Firstbit_LP = 0;
            Secondbit_LP = 1;
        elseif   (R<3)&&(R>=0)&&(I<-3)      % 3rd Quadrant around HP point (3,-3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 1;
        elseif  (R>=3)&&(I<-3)             % 4th Quadrant around HP point (3,-3) at origin
            Firstbit_LP = 1;
            Secondbit_LP = 0;
        end



        Firstbit_HPsymbol = [Firstbit_HPsymbol, Firstbit_HP];
        Secondbit_HPsymbol = [Secondbit_HPsymbol, Secondbit_HP];
        Firstbit_LPsymbol = [Firstbit_LPsymbol, Firstbit_LP];
        Secondbit_LPsymbol = [Secondbit_LPsymbol, Secondbit_LP];
    end

    %  HP stream demodulated symbols saved in a matrix column wise
    %  LP stream demodulated symbols saved in a matrix column wise
    HP_Demod_Symbols= vertcat(Firstbit_HPsymbol,Secondbit_HPsymbol);  
    LP_Demod_Symbols= vertcat(Firstbit_LPsymbol,Secondbit_LPsymbol);  

    [o,p] = size(HP_Demod_Symbols);
    q = o*p;

    %  HP stream demodulated bits saved in a matrix
    %  LP stream demodulated bits saved in a matrix
    HP_Demod_Signal = reshape(HP_Demod_Symbols,1,q);     
    LP_Demod_Signal = reshape(LP_Demod_Symbols,1,q);     

    [t,u] = size(HP_Demod_Signal);
    v=t*u;

    % check if no of bits are divisible by 3 or not... if not discard last 
    % bit to make it completely divisble by 3
    if mod(v,3) == 0  
    else
        HP_Demod_Signal(end) = [];
        LP_Demod_Signal(end) = [];
    end

    tbdepth = 1;

    % Channel decoding
    HP_Channel_Decoded_Signal = vitdec(HP_Demod_Signal, trellis,tbdepth,...
        'trunc', 'hard');
    LP_Channel_Decoded_Signal = vitdec(LP_Demod_Signal, trellis,tbdepth,...
        'trunc', 'hard');

    % check whether channel decoded bits i.e output of channel decoder and 
    % the source coded bits i.e input to channel encoder are equal or not!
    [HPChannelDecodSize_Row,HPChannelDecodSize_Col]= size(...
        HP_Channel_Decoded_Signal);
    Size_ChannelDecode_Signal=HPChannelDecodSize_Row*HPChannelDecodSize_Col;
    [HP_SourceCoded_Row,HP_SourceCoded_Col] = size(SourceCoded_HPSignal);
    [LP_SourceCoded_Row,LP_SourceCoded_Col] = size(SourceCoded_LPSignal);
    Extra_Bits_HP=[]; Extra_Bits_LP=[];
    Additional_Bits_HP=[]; Additional_Bits_LP=[];
    % check for HP stream
    if Size_ChannelDecode_Signal==HP_SourceCoded_Row                            %  check if channel decoded and source encoded bits are equal then do nothing
        HP_Channel_Decoded_Signal_Complete=HP_Channel_Decoded_Signal;
    elseif Size_ChannelDecode_Signal>HP_SourceCoded_Row                         %  check if channel decoded bits are greater then remove the extra bits from received signal
        HP_Channel_Decoded_Signal(Size_ChannelDecode_Signal+1:end) = [];
        HP_Channel_Decoded_Signal_Complete=HP_Channel_Decoded_Signal;
    elseif Size_ChannelDecode_Signal<HP_SourceCoded_Row                         %  check if channel decoded bits are less then append the additional bits from the original signal
        for k=Size_ChannelDecode_Signal+1:HP_SourceCoded_Row
            Extra_Bits_HP= SourceCoded_HPSignal(k,1);
            Additional_Bits_HP=[Additional_Bits_HP, Extra_Bits_HP];
        end
        HP_Channel_Decoded_Signal_Complete=horzcat(HP_Channel_Decoded_Signal,Additional_Bits_HP);
    end
    
    %   Check for LP stream:
    %       - If demodulated bits are equal, do nothing
    %       - If demodulated bits are greater, remove the extra bits from 
    %             the received signal
    %       - If demodulated bits are less than, append the additional bits 
    %             from the original signal
    %
    if Size_ChannelDecode_Signal==LP_SourceCoded_Row                            
        LP_Channel_Decoded_Signal_Complete=LP_Channel_Decoded_Signal;
    elseif Size_ChannelDecode_Signal>LP_SourceCoded_Row                         
        LP_Channel_Decoded_Signal(Size_ChannelDecode_Signal+1:end) = [];
        LP_Channel_Decoded_Signal_Complete=LP_Channel_Decoded_Signal;
    elseif Size_ChannelDecode_Signal<LP_SourceCoded_Row                         
        for k=Size_ChannelDecode_Signal+1:LP_SourceCoded_Row
            Extra_Bits_LP= SourceCoded_LPSignal(k,1);
            Additional_Bits_LP=[Additional_Bits_LP, Extra_Bits_LP];
        end
        LP_Channel_Decoded_Signal_Complete=horzcat(LP_Channel_Decoded_Signal,Additional_Bits_LP);
    end


    %  Computation of BER for HP stream
    [y,z] = size(HP_Channel_Decoded_Signal_Complete);
%     [y,z] = size(SourceCoded_HPSignal);
    E=[];

    for i=1:z
        % Compare the input of channel coder block and output of the 
        % channel decoder block; if the bits are the same, i.e correctly 
        % received, write '0', otherwise write '1'
        if  HP_Channel_Decoded_Signal_Complete(1,i) == ...
                SourceCoded_HPSignal(i,1)  
            s=0;                                   
        else
            s=1;
        end

        % Save the result in a matrix
        E = [E,s]; 
    end

    % Count the # of 1's in the matrix to know how many bits are in error
    count_HP = length(E(E==1));  
    BER_QPSK_HP = count_HP / (y*z);
    BER_HP=[BER_HP,BER_QPSK_HP];

    %  Computation of BER for LP stream
    [LProw,LPCol] = size(LP_Channel_Decoded_Signal_Complete);
%     [LProw,LPCol] = size(SourceCoded_LPSignal);
    F = [];

    for i=1:LPCol
        % Compares the input of channel coder block and output of the 
        % channel decoder block; if bits are the same, i.e correctly 
        % received, write a '0', otherwise write '1'       
        if  LP_Channel_Decoded_Signal_Complete(1,i) ==  ...
                SourceCoded_LPSignal(i,1)  
            s=0;                                   
        else
            s=1;
        end

        F = [F,s]; % save the result in a matrix
    end

    % Count the # of 1s in the matrix to know how many bits are in error
    count_LP = length(F(F==1));  
    BER_QPSK_LP = count_LP / (LProw*LPCol);
    BER_LP=[BER_LP,BER_QPSK_LP];

    %% experiment 2 begin

    % Source decoding
    HP_Source_Decoded_Signal = huffmandeco( ...
        HP_Channel_Decoded_Signal_Complete, dict_hp);
    LP_Source_Decoded_Signal = huffmandeco( ...
        LP_Channel_Decoded_Signal_Complete, dict_lp);

    % Reassemble decoded images
    size_delta_hp = numel(file_gray) - length(HP_Source_Decoded_Signal);
    size_delta_lp = numel(file_gray) - length(LP_Source_Decoded_Signal);
    
    HP_Source_Decoded_Signal_tmp = [HP_Source_Decoded_Signal zeros(1, size_delta_hp)];
    LP_Source_Decoded_Signal_tmp = [LP_Source_Decoded_Signal  zeros(1, size_delta_lp)];

    [p1,q1] = size(HP_Source_Decoded_Signal_tmp);
    [p2,q2] = size(LP_Source_Decoded_Signal_tmp);

    if ( q1 > q2)
        HP_Source_Decoded_Signal_tmp(q2+1:end) = [];
    elseif (q2>q1)
        LP_Source_Decoded_Signal_tmp(q1+1:end) = [];
    end

     Combined_Source_Decoded_Signal = ...
         LP_Source_Decoded_Signal_tmp + HP_Source_Decoded_Signal_tmp;

%     Combined_Source_Decoded_Signal = ...
%         [HP_Source_Decoded_Signal  zeros(1, size_delta_hp)] + ...
%         [LP_Source_Decoded_Signal  zeros(1, size_delta_lp)];
%     Combined_Source_Decoded_Signal = HP_Source_Decoded_Signal + LP_Source_Decoded_Signal; 


    % Reshape image and IDCT2
    NN = floor(sqrt(length(Combined_Source_Decoded_Signal)));
    DS_temp = reshape(Combined_Source_Decoded_Signal, NN, []);
    RX_Decoded_Image = rescale(idct2(DS_temp));

        % Display original image, and compare with received HP layer, LP layer, and
        % combined (HP + LP)
        
        
        figure
        hold on;
        
        subplot(221)
        imshow(file_gray);
        title("Original: SNR = "  + num2str(index) + " dB");
        subplot(222)
        imshow(rescale(idct2(reshape(...
            HP_Source_Decoded_Signal_tmp, NN, []))));
        title("HP: SNR = "  + num2str(index) + " dB");
        subplot(223)
        imshow(rescale(idct2(reshape(...
            LP_Source_Decoded_Signal_tmp, NN, []))));
        title("LP: SNR = "  + num2str(index) + " dB");
        subplot(224)
        imshow(RX_Decoded_Image);
        title("Combined (HP + LP): SNR = " + num2str(index) + " dB");
        hold off;

    %% experiment 2 end
end


BER_LP_dB=10*log(BER_LP);
BER_HP_dB=10*log(BER_HP);
% figure;
% subplot(2,1,1);plot(SNR,BER_HP_dB); title('HP Stream BER Curve');
% subplot(2,1,2);plot(SNR,BER_LP_dB); title('LP Stream BER Curve');
figure;
hold on;
% subplot(2,1,1);
semilogy(SNR,BER_HP, 'Displayname', 'HP Stream');
title('HP & LP Stream BER Curve');xlabel('SNR'),ylabel('BER');
% axis([0 14 0 05]);
%figure;
% subplot(2,1,2);
semilogy(SNR,BER_LP,'Displayname', 'LP Stream');
% axis([0 20 0 0.2]);
legend;
hold off;
grid on;

