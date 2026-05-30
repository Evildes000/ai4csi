clc;
clear;
close all;

%% =========================================
% OFDM Parameters

Nfft = 64;                 % FFT size
Ncp  = 16;                 % Cyclic Prefix
M    = 4;                  % QPSK
Fs   = 20e6;               % Sampling frequency (20 MHz)

numSubcarrier = Nfft;

%% =========================================
% Dataset Parameters

distance_list = 1:2:40;      % 1~10 meters
num_packet = 2000;

%% =========================================
% Physical Constants

c = 3e8;                   % Speed of light
fc = 5.8e9;                % Carrier frequency
lambda = c/fc;

delta_f = Fs/Nfft;         % Subcarrier spacing

%% =========================================
% Generate Pilot

pilotBits = randi([0 1], Nfft*2, 1);

pilotSymbol = qammod( ...
    bi2de(reshape(pilotBits,2,[]).','left-msb'), ...
    M, ...
    'UnitAveragePower', true);

pilotSymbol = pilotSymbol(:);

%% =========================================
% Dataset Structure

CSI_Dataset = struct();

%% =========================================
% Main Loop

for d = distance_list

    fprintf('Processing Distance = %d m\n', d);

    % Save CSI of current distance
    CSI_temp = complex(zeros(num_packet, Nfft));

    %% =====================================
    % Propagation delay

    tau = d / c;

    %% =====================================
    % Path loss (Free Space Path Loss)

    pathLoss_dB = 20*log10(4*pi*d/lambda);

    pathGain = 10^(-pathLoss_dB/20);

    %% =====================================
    % Subcarrier index

    k = (0:Nfft-1).';

    for pkt = 1:num_packet

        %% =================================
        % Single-path Rayleigh fading

        h = (randn + 1j*randn)/sqrt(2);

        %% =================================
        % Frequency-domain channel
        %
        % H(k) = h * exp(-j2πkΔfτ)

        H = h .* exp(-1j*2*pi*k*delta_f*tau);

        % Add path loss
        H = H * pathGain;

        %% =================================
        % Transmit OFDM symbols

        txFreq = pilotSymbol;

        %% =================================
        % Received signal in frequency domain

        rxFreq = txFreq .* H;

        %% =================================
        % Add AWGN noise

        SNR_dB = 25;

        signalPower = mean(abs(rxFreq).^2);

        noisePower = signalPower / (10^(SNR_dB/10));

        noise = sqrt(noisePower/2) * ...
            (randn(size(rxFreq)) + 1j*randn(size(rxFreq)));

        rxFreq = rxFreq + noise;

        %% =================================
        % CSI Estimation

        CSI = rxFreq ./ txFreq;

        %% =================================
        % Save CSI

        CSI_temp(pkt,:) = CSI.';

    end

    %% =====================================
    % Save to dataset

    field_name = sprintf('distance_%d', d);

    CSI_Dataset.(field_name) = CSI_temp;

end

%% =========================================
% Save dataset

save('CSI_Dataset.mat', 'CSI_Dataset');

disp('CSI Dataset Saved Successfully');

%% =========================================
% Visualization Example

example_distance = 5;

field_name = sprintf('distance_%d', example_distance);

CSI_example = CSI_Dataset.(field_name);

CSI_one_packet = CSI_example(1,:);

figure;

subplot(2,1,1);
plot(abs(CSI_one_packet), 'LineWidth', 2);
grid on;
xlabel('Subcarrier Index');
ylabel('Amplitude');
title(['Amplitude of CSI (Distance = ', num2str(example_distance), ' m)']);

subplot(2,1,2);
plot(unwrap(angle(CSI_one_packet)), 'LineWidth', 2);
grid on;
xlabel('Subcarrier Index');
ylabel('Phase (rad)');
title(['Phase of CSI (Distance = ', num2str(example_distance), ' m)']);