clear;
clc;
% paramters:    lambdam1    lambda2    lambda3    gamma
% arts          2^-8        2^-2       2^-6      10   
% birds         2^-10       2^0        2^-10      1   
% cal500        2^-10       2^-7       2^-8       1000      
% flags         2^0         2^-8       2^1        1
% genbase       2^-3        2^-2       2^2        0.1
% medical       2^-6        2^-3       2^-2       1
% rcv1          2^-1        2^-4       2^1        1000
% rcv2          2^-5        2^-8       2^1        1000
% rcv3          2^-10       2^-5       2^1        1000
% rcv4          2^-9        2^-4       2^0        1000
% rcv5          2^-8        2^-3       2^1        1000
% slashdot      2^-9        2^-7       2^-7       10
% yeast         2^-2        2^-8       2^-1       10
dataset = 'yeast'; % 2^-4£¬2^2, 0.1
cd('data');
    eval(['load ', dataset]);
    eval(['load ', dataset, '_search']);
cd('..');
% features = zscore(features);
num_instance = size(features, 1);
lastcol = ones(num_instance,1);
features = [features, lastcol];
% Paramters search range0
lambda1_range = 2.^(-10:-1); % label correlation1
lambda2_range = 2.^(-10:-6); % sample similar 
lambda3_range = 2.^(-1:-1); % sparsity
gamma_range = 10.^(1:1);
opt_params.maxIter = 100;
opt_params.neg = 10;
opt_params.minimumLossMargin = 0.0001;
BestParameter = opt_params;
index = 0;
total = length(lambda1_range)*length(lambda2_range)*length(lambda3_range)*length(gamma_range);
% Result
BestResult = ones(1,5);
Result = zeros(1, 7);
Allres = zeros(total, 9);
for i=1:length(lambda1_range)
    for j=1:length(lambda2_range)
        for l=1:length(lambda3_range)
            for k=1:length(gamma_range)
               index = index + 1;
               opt_params.lambda1 = lambda1_range(i); % label correlation
               opt_params.lambda2 = lambda2_range(j); % label correlation
               opt_params.lambda3 = lambda3_range(l);  % sparsity
               opt_params.gamma = gamma_range(k);
               fprintf('%dth / %d --search params, lambda1 = %f, lambda2 = %f, lambda3 = %f, gamma = %f %s \n',index, total, opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.gamma, datestr(now));
               temp_result = zeros(5, 7);
               for rep=1:5
        %          fprintf('===============%d %s ============== \n', rep, datestr(now));
                   testIdx = find(indices == rep);
                   trainIdx = setdiff(find(indices),testIdx);
                   test_feature = features(testIdx,:);
                   test_target = labels(testIdx,:);
                   train_feature = features(trainIdx,:);
                   train_target = labels(trainIdx,:);
                   % Train model
                   [W]  = JFSC(train_feature, train_target, opt_params);
                   % Prediction
                   [pre_labels, pre_dis , res_once] = JFSC_predict(W, test_feature, test_target);
                   temp_result(rep, :) = res_once;
                   Allres(index, :) = [opt_params.lambda1, opt_params.lambda2, opt_params.lambda3, opt_params.gamma, res_once(1:5)];
        %          fprintf('=============== %s ============== \n', datestr(now));
               end
               Result(1, :) = mean(temp_result, 1);
               r = IsBetterThanBefore(BestResult,Result);
               if r == 1
                   BestResult = Result;
                   fprintf('BestResult: %.4f, %.4f, %.4f, %.4f, %.4f \n',Result(1,1), Result(1,2), Result(1,3), Result(1,4), Result(1,5));
                   BestParameter = opt_params;
               end
            end
        end
    end
end

function r = IsBetterThanBefore(Result,CurrentResult)
    a = CurrentResult(1,1) + CurrentResult(1,2)  + CurrentResult(1,3) + CurrentResult(1,4) - CurrentResult(1,5);
    b = Result(1,1) + Result(1,2) + Result(1,3) + Result(1,4) - Result(1,5);
    if a < b
        r =1;
    else
        r = 0;
    end
end