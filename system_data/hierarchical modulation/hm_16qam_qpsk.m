% Clear everything
clear all;
close all;
clc;
tic;

% Set the simulation parameter
EbN0_dB = 0:5:30;
Frames_Num = 200;
MAX_Iter_Num = 5;

% LBP
bitError_LBP = zeros(1, length(EbN0_dB));
BER_LBP = zeros(1, length(EbN0_dB));
iterNumTotal_LBP = zeros(1, length(EbN0_dB));

INFO_LENGTH = 4096;
RATE = 4/5;
SIZE_M = 512;
NORM_FACTOR = 0.75;

% Generate LDPC matrices for each layer
H = ccsdscheckmatrix(SIZE_M, RATE);
G = ccsdsgeneratematrix(H1, SIZE_M, RATE);

for nEbN0 = 1:length(EbN0_dB)
    fprintf('%2.2f', EbN0_dB(nEbN0));
    fprintf('dB,');
    fprintf('m = %d,codenum = ', MAX_Iter_Num);

    SNR_per_bit = 10^(EbN0_dB(nEbN0) / 10);
    SNR_per_sym = SNR_per_bit * (4 + 2); % 16-QAM has 4 bits per symbol, QPSK has 2 bits per symbol
    N0 = 1 / (SNR_per_sym * RATE);
    sigma = sqrt(N0 / 2);

    for nF = 1:Frames_Num
        fprintf('%d,', nF);

        % Encode
        message = randi([0 1], 1, INFO_LENGTH);
        
        % Split message for two layers
        message_layer1 = message(1:2:end); % QAM Layer
        message_layer2 = message(2:2:end); % QPSK Layer
        
        % LDPC Encode each layer
        encodeData_layer1 = mod(message_layer1 * G1, 2);
        encodeData_layer2 = mod(message_layer2 * G2, 2);

        % 16-QAM Modulation for Layer 1
        M1 = 16;
        k1 = log2(M1);
        numSymbols1 = length(encodeData_layer1) / k1;
        symbols1 = bi2de(reshape(encodeData_layer1, k1, numSymbols1).', 'left-msb');
        layer1_signal = qammod(symbols1, M1, 'UnitAveragePower', true);

        % QPSK Modulation for Layer 2
        M2 = 4;
        k2 = log2(M2);
        numSymbols2 = length(encodeData_layer2) / k2;
        symbols2 = bi2de(reshape(encodeData_layer2, k2, numSymbols2).', 'left-msb');
        layer2_signal = pskmod(symbols2, M2, pi/4);

        % Make the lengths of layer1_signal and layer2_signal the same
        minLength = min(length(layer1_signal), length(layer2_signal));
        layer1_signal = layer1_signal(1:minLength);
        layer2_signal = layer2_signal(1:minLength);

        % Superposition of both layers
        transmitSignal = layer1_signal + layer2_signal;

        % Rayleigh Channel Generation
        h = (randn(size(transmitSignal)) + 1i * randn(size(transmitSignal))) / sqrt(2);

        % AWGN Channel
        receiveSignal = h .* transmitSignal + sigma * (randn(size(transmitSignal)) + 1i * randn(size(transmitSignal)));

        % Channel Equalization
        Est_err_para = 0;
        Est_err = Est_err_para * abs(h);
        hEst = h + Est_err;
        equalizedSignal = receiveSignal ./ hEst;

        % Serial Interference Cancellation (SIC)
        % Demodulate Layer 1 (16-QAM)
        layer1_demodSymbols = qamdemod(equalizedSignal, M1, 'UnitAveragePower', true);
        layer1_receivedBits = de2bi(layer1_demodSymbols, k1, 'left-msb');
        layer1_receivedBits = layer1_receivedBits(:).';

        % Decode Layer 1 Bits using LDPC
        [iter_Num_LBP_layer1, recoverData_LBP_layer1] = LBP(layer1_receivedBits, H1, MAX_Iter_Num, NORM_FACTOR);

        % Re-modulate Layer 1 Bits
        symbols1_remod = bi2de(reshape(recoverData_LBP_layer1, k1, []).', 'left-msb');
        layer1_remod_signal = qammod(symbols1_remod, M1, 'UnitAveragePower', true);

        % Subtract Layer 1 signal from received signal
        remainingSignal = receiveSignal - hEst .* layer1_remod_signal;

        % Demodulate Layer 2 (QPSK)
        layer2_demodSymbols = pskdemod(remainingSignal ./ hEst, M2, pi/4);
        layer2_receivedBits = de2bi(layer2_demodSymbols, k2, 'left-msb');
        layer2_receivedBits = layer2_receivedBits(:).';

        % Decode Layer 2 Bits using LDPC
        [iter_Num_LBP_layer2, recoverData_LBP_layer2] = LBP(layer2_receivedBits, H2, MAX_Iter_Num, NORM_FACTOR);

        % Combine recovered bits from both layers
        recoverData_LBP = zeros(1, length(message));
        recoverData_LBP(1:2:end) = recoverData_LBP_layer1(1:minLength*k1);
        recoverData_LBP(2:2:end) = recoverData_LBP_layer2(1:minLength*k2);

        % Bit error, BER, iterNumTotal calculation
        bitError_LBP(nEbN0) = bitError_LBP(nEbN0) + sum(abs(message - recoverData_LBP));
        iterNumTotal_LBP(nEbN0) = iterNumTotal_LBP(nEbN0) + iter_Num_LBP_layer1 + iter_Num_LBP_layer2;
        if (nF == Frames_Num)
            BER_LBP(nEbN0) = bitError_LBP(nEbN0) / (nF * length(message));
            break;
        end
    end  % End for 'nF'
    fprintf('\n');
end  % End for 'nEbN0'

semilogy(EbN0_dB, BER_LBP, '+b-');
xlabel('Eb/N0(dB)');
ylabel('BER');
grid on;
toc;
