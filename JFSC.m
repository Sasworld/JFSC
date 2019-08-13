function [model_LLSF] = JFSC(X, Y, optmParameter)
    
   %% optimization parameters
    lambda1          = optmParameter.lambda1;
    lambda2          = optmParameter.lambda2;
    lambda3          = optmParameter.lambda3;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

   %% initializtion
    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_s_1 = W_s;
    
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );
   
    S = get_S(X, Y);
    A = [];
    for a = 1:size(S, 2)
        A = [A, lambda2 .* sum(S{a}, 2)];
    end
    
    iter    = 1;
    oldloss = 0;
    D_X = diag(1 ./ sqrt(sum((X * W_s - Y).^2, 2)));
    Lip = sqrt(3 * (norm(X'*D_X*X)^2 + norm(lambda1 * R)^2 + max(max(A.^2))));

    bk = 1;
    bk_1 = 1; 
    
   %% proximal gradient
    while iter <= maxIter
       D_X = diag(1 ./ sqrt(sum((X * W_s - Y).^2, 2)));
       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);
       Gw_s_k = W_s_k - 1/Lip * (X'*D_X*X*W_s_k - X'*D_X*Y + lambda1 * W_s_k * R + A .* W_s_k);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,lambda3/Lip);
       
       predictionLoss = trace((X*W_s - Y)'*D_X*(X*W_s - Y));
       correlation     = trace(W_s*R*W_s');
       sparsity = sum(sum(abs(W_s)), 2);
       sample = 0;
       for a = 1:size(S, 2)
           sample = sample + W_s(:, a)' * S{a} * W_s(:, a);
       end
%        sparsity    = sum(sum(W_s~=0));
       totalloss = predictionLoss / 2 + lambda1 * correlation / 2 + lambda2 * sample / 2 + lambda3 * sparsity;
%        fprintf('=============== predictionLoss: %f ================ \n', predictionLoss);
%        fprintf('=============== correlation: %f ================ \n', correlation);
%        fprintf('=============== sparsity: %f ================ \n', sparsity);
%        fprintf('=============== %d %f ================ \n', iter, totalloss);
       if abs(oldloss - totalloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    model_LLSF = W_s;
end


%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end
