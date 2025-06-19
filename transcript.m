clear
clc
close all


%% In[1]:

n = 11;  % number of points
x = linspace(0, 1, n).^2; x = min(x, flip(x));  % arbitrary signal
L = 5;  % number of points to average
h = ones(L, 1)/L;
y = conv(h, x);

figure
hold on
plot(x, 'o-', 'DisplayName', '$x$')
plot(h, 'o-', 'DisplayName', '$h$')
plot(y, 'o-', 'DisplayName', '$y=h*x$')
hl = legend;
set(hl, 'Interpreter', 'latex')


%% In[2]:

n = 500;  % number of points
n_T = 3;  % number of periods

t = (0:n-1)/n*n_T*2*pi;
x = mod(t/pi + 1, 2) - 1;

figure
hold on
plot(t, x, 'DisplayName', 'sawtooth')

for n_components = [1, 2, 5, 10]
    x = 0;
    for k = 1:n_components
        x = x - 2/pi*(-1)^k/k*sin(k*t);
    end
    plot(t, x, 'DisplayName', sprintf('%i components', k))
end

legend


%% In[3]:

n = 512;  % number of points
fs = 16e3;  % sampling frequency

f1 = 2000;  % frequency of the first component
f2 = 4000;  % frequency of the second component

t = (0:n-1)/fs;  % time axis
x = sin(2*pi*f1*t) + sin(2*pi*f2*t);  % time-domain signal
X = fft(x);  % DFT
f = (0:n-1)/n*fs;  % frequency axis; see details further below

figure
subplot(2, 1, 1)
plot(t, x)
title('Time domain')
xlabel('Time (s)')
subplot(2, 1, 2)
plot(f, abs(X))  % we plot the magnitude as X is complex
title('Frequency domain')
xlabel('Frequency (Hz)')


%% In[4]:

n = 512;
t = (0:n-1) - floor(n/2);
f = (0:n-1)*fs/n; f(f>=fs/2) = f(f>=fs/2) - fs; f = sort(f);

n_rows = 8;
x = zeros(n, n_rows);

% dirac
x(:, 1) = t == 0;
% constant
x(:, 2) = ones(1, n);
% rectangle
x(:, 3) = abs(t) < n*0.025;
% sinc
x(:, 4) = sinc(t*0.1);
% comb
x(:, 5) = mod(t, floor(n/32)) == 0;
% sine
x(:, 6) = sin(2*pi*t*0.05);
% cosine
x(:, 7) = cos(2*pi*t*0.05);
% sawtooth
x(:, 8) = mod(t*0.1 + 1, 2) - 1;

labels = {
    'dirac';
    'constant';
    'rectangle',;
    'sinc';
    'comb';
    'sine';
    'cosine';
    'sawtooth';
};

figure
for i = 1:n_rows
    X = fft(x(:, i)); X = fftshift(X);
    subplot(n_rows, 2, 2*i-1)
    plot(t, x(:, i))
    if i ~= n_rows
        set(gca,'XTickLabel',[]);
    end
    subplot(n_rows, 2, 2*i)
    plot(f, abs(X))    
    if i ~= n_rows
        set(gca,'XTickLabel',[]);
    end
end
subplot(n_rows, 2, 1)
title('Time domain')
subplot(n_rows, 2, 2)
title('Frequency domain')


%% In[5]:

n = 128;  % number of points

figure

% first property: F(x*y)=F(x)F(y)
x = randn(1, n);
y = randn(1, n);
z = conv(x, y);

X = fft(x, length(z));  % forcing the FFT output to be same length as z
Y = fft(y, length(z));  % forcing the FFT output to be same length as z

Z1 = fft(z);
Z2 = X.*Y;

subplot(2, 1, 1)
hold on
plot(abs(Z1), 'DisplayName', '$\mathcal{F}(x*y)$')
plot(abs(Z2), 'DisplayName', '$\mathcal{F}(x)\mathcal{F}(y)$')
hl = legend;
set(hl, 'Interpreter', 'latex')

% second property: F(xy)=F(x)*F(y)
% this one is a bit trickier as we need to flip the FFTs before convolving
% we also need to filter out all the extra frequencies resulting from the convolution in the frequency domain
x = sin(2*pi*(0:n-1)*0.3);  % using random noise here does not give perfect result
y = sin(2*pi*(0:n-1)*0.1);  % using random noise here does not give perfect result
z = x.*y;

X = fft(x);
Y = fft(y);
X = fftshift(X);  % flip before convolving
Y = fftshift(Y);  % flip before convolving

Z1 = fft(z);
Z1 = fftshift(Z1);
Z2 = conv(X, Y)/n;
Z2 = Z2(floor(n/2)+1:end-floor(n/2)+1);  % discard extra frequencies created from the convolution

subplot(2, 1, 2)
hold on
plot(abs(Z1), 'DisplayName', '$\mathcal{F}(xy)$')
plot(abs(Z2), 'DisplayName', '$\frac{1}{N}\mathcal{F}(x)*\mathcal{F}(y)$')
hl = legend;
set(hl, 'Interpreter', 'latex')


%% In[6]:

f0 = 100;  % sinusoid frequency
T = 2e-2;  % sinusoid duration in seconds

% first create a sinusoid with a fine time step; this will represent the continuous signal
fs_hi = 8e3;  % high sampling frequency
t_cont = 0:1/fs_hi:T;  % fine time vector with time step 1/fs
x_cont = sin(2*pi*f0*t_cont);  % this represents the continuous signal

figure

% now let's create a coarse digital signals for different low sampling frequencies
fs_lo = [1000, 500, 200];
for i = 1:length(fs_lo)
    subplot(1, 3, i)
    hold on
    plot(t_cont, x_cont)
    t_coarse = 0:1/fs_lo(i):T;
    x_coarse = sin(2*pi*f0*t_coarse);
    stem(t_coarse, x_coarse, 'k')
    ht = title(sprintf('$f_s=%i$ Hz', fs_lo(i)));
    set(ht, 'Interpreter', 'latex')
end


%% In[7]:

figure
hold on

f0 = 100;
f1 = 80;

x_cont = cos(2*pi*f0*t_cont);
plot(t_cont, x_cont)

x_cont = cos(2*pi*f1*t_cont);
plot(t_cont, x_cont)

fs_lo = 180;
t_coarse = 0:1/fs_lo:T;
x_coarse = cos(2*pi*f0*t_coarse);
stem(t_coarse, x_coarse, 'k')

ht = title(sprintf('$f_s=%i$ Hz', fs_lo));
set(ht, 'Interpreter', 'latex')
hl = legend(sprintf('$f_0=%i$ Hz', f0), sprintf('$f_1=%i$ Hz', f1));
set(hl, 'Interpreter', 'latex')


%% In[8]:

fs = 4e3;  % sampling frequency
f_mod = 1;  % modulation frequency
f_delta = 200;  % modulation depth
f0 = 800;  % carrier frequency
T = 5;  % signal duration
t = 0:1/fs:T;  % time vector
x = sin(2*pi*t*f0 + f_delta/f_mod*sin(2*pi*t*f_mod));

figure

subplot(3, 1, 1)
plot(t, x)
title('Raw time-domain signal')
xlabel('Time (s)')
ylabel('Amplitude')

X = fft(x); X = X(1:floor(length(X)/2)+1);
f = (0:length(X)-1)*fs/length(x);

subplot(3, 1, 2)
plot(f, abs(X));
title('FFT')
xlabel('Frequency (Hz)')
ylabel('Magnitude')

subplot(3, 1, 3)
stft(x, fs, 'FrequencyRange', 'onesided');


%% In[9]:

figure

X = fft(x); X = X(1:floor(length(X)/2)+1);
f = (0:length(X)-1)*fs/length(x);

subplot(2, 1, 1)
plot(f, abs(X))
title('FFT (ugly)')
xlabel('Frequency (Hz)')
ylabel('Magnitude')

[s,f,t] = stft(x, fs, 'FrequencyRange', 'onesided');
LTAS = mean(abs(s), 2);

subplot(2, 1, 2)
plot(f, LTAS)
title('LTAS (clean)')
xlabel('Frequency (Hz)')
ylabel('Magnitude')


%% In[10]:

% an arbitrary noisy input signal
n = 512;  % signal length
x = 0.5*randn(1, n) + cos(2*pi*(0:n-1)*0.01) + cos(2*pi*(0:n-1)*0.005);

% moving average filter
L = 10;  % number of points to average
b = ones(1, L)/L;  % feedforward coefficients
a = 1;  % feedback coefficients
y1 = filter(b, a, x);

% exponential smoothing filter
alpha = 0.9;
b = 1-alpha;  % feedforward coefficients
a = [1, -alpha];  % feedback coefficients
y2 = filter(b, a, x);

figure
hold on
plot(x, 'DisplayName', 'input signal')
plot(y1, 'DisplayName', 'moving average')
plot(y2, 'DisplayName', 'exponential smoothing')
legend


%% In[11]:

n = 512;  % number of frequency points to evaluate

% moving average filter
L = 10;  % number of points to average
b = ones(1, L)/L;  % feedforward coefficients
a = 1;  % feedback coefficients
[h1, w1] = freqz(b, a, n);

% exponential smoothing filter
alpha = 0.9;
b = 1-alpha;  % feedforward coefficients
a = [1, -alpha];  % feedback coefficients
[h2, w2] = freqz(b, a, n);

figure
hold on
plot(w1, abs(h1), 'DisplayName', 'moving average')
plot(w2, abs(h2), 'DisplayName', 'exponential smoothing')
legend
