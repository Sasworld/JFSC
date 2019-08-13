function macrof1 = MacroF1(Pre_Labels,test_target)
%Computing the Macro_AUC
%Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
    Pre_Labels(Pre_Labels==0)=-1;
    test_target(test_target==0)=-1;

    [num_class,num_instance]=size(Pre_Labels);
	num_P_instance = zeros(num_class,1);%number of positive instance for every label
    num_N_instance = zeros(num_class,1);
    num_P_pre = zeros(num_class,1);%number of positive instance for every label
    num_N_pre = zeros(num_class,1);
    fm = zeros(num_class,1);
    for i = 1:num_class
        num_P_instance(i,1) = sum(test_target(i,:) == 1); % 标记i的正例样本数量
        num_N_instance(i,1) = num_instance - num_P_instance(i,1);
        num_P_pre(i,1) = sum(Pre_Labels(i,:) == 1);
        num_N_pre(i,1) = num_instance - num_P_pre(i,1);

        num_P = sum(Pre_Labels(i,:) == 1);
        num_N = num_instance - num_P;
        pre=Pre_Labels(i,:);
        pre0=pre;
        instance=test_target(i,:);
        pre0(pre0==-1)=0; % 将预测的反例值设为0（原本为-1）
        TP=sum(pre0==instance); %真正例
        pre0=pre;
        pre0(pre0==1)=0;
        TN=sum(pre0==instance);
        FP=num_P-TP;
        FN=num_N-TN;
        if TP+FP+FN==0
            fm(i,1) = 1;
        else
			fm(i,1)=2*TP/(2*TP+FN+FP);
        end
    end
	macrof1 = sum(fm)/num_class;
end

