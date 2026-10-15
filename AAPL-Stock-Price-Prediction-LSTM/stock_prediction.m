clear; close all; clc

file = "aapl_us_d.csv";
closeCol = 5;
numTimeSteps = 20;
trainRatio = 0.8;

T = readtable(file);
price = fillmissing(T{:,closeCol}(:),"linear");

minP = min(price); maxP = max(price);
priceNorm = (price - minP) / (maxP - minP);

[X, Y] = makeSeq(priceNorm, numTimeSteps);
n = numel(Y);
if n < 10, error("Not enough sequences. Reduce numTimeSteps. n=%d", n); end

nTrain = floor(trainRatio*n);
XTrain = X(1:nTrain); YTrain = Y(1:nTrain);
XTest  = X(nTrain+1:end); YTest  = Y(nTrain+1:end);

layers = [
    sequenceInputLayer(1)
    lstmLayer(100,"OutputMode","last")
    fullyConnectedLayer(1)
];

opts = trainingOptions("adam", ...
    "MaxEpochs",100, ...
    "MiniBatchSize",16, ...
    "InitialLearnRate",0.005, ...
    "GradientThreshold",1, ...
    "Shuffle","never", ...
    "Verbose",true, ...
    "Plots","training-progress");

net = trainnet(XTrain, YTrain, layers, "mse", opts);

%% --- FIX: stack cell sequences for dlnetwork/predict ---
XTrainA = cat(3, XTrain{:});      % win x 1 x NTrain
XTestA  = cat(3, XTest{:});       % win x 1 x NTest

YPredTestNorm = squeeze(predict(net, XTestA));
YTestTrue     = denorm(YTest, minP, maxP);
YTestPred     = denorm(YPredTestNorm, minP, maxP);

rmseTest = sqrt(mean((YTestPred - YTestTrue).^2));
fprintf("Test RMSE: %.6f\n", rmseTest);

figure
plot(YTestTrue,'b','LineWidth',1.5); hold on
plot(YTestPred,'r--','LineWidth',1.5);
grid on; legend("Actual","Predicted","Location","best")
title(sprintf("Test Set | RMSE=%.4f", rmseTest))
xlabel("Test sample index"); ylabel("Close Price")

%% Next-day prediction (numeric array, not cell)
lastSequence = reshape(priceNorm(end-numTimeSteps+1:end), [], 1); % win x 1
lastSeqA = reshape(lastSequence, numTimeSteps, 1, 1);            % win x 1 x 1

futurePredNorm = squeeze(predict(net, lastSeqA));
futurePred = denorm(futurePredNorm, minP, maxP);
fprintf("Predicted Next Day Price: %.4f\n", futurePred);

%% Helpers
function [X, Y] = makeSeq(v, win)
n = numel(v) - win;
X = cell(n,1);
Y = zeros(n,1);
for k = 1:n
    X{k} = reshape(v(k:k+win-1), [], 1); % win x 1
    Y(k) = v(k+win);
end
end

function y = denorm(yNorm, mn, mx)
y = yNorm*(mx-mn) + mn;
end