function [edu_dis] = sample_similary(M, neg)
    % ����M����֮���ŷʽ����
    [~, col] = size(M');
    tempA = repmat(sum(M.^2,2),1,col);
    tempB = tempA';
    tempC = M*M';
    edu_dis = tempA+tempB-2*tempC;
    temp1 = sort(edu_dis, 2);
    krd_value = temp1(:,neg + 1);
    temp2 = edu_dis;
    temp2 = temp2 - krd_value;
    edu_dis(temp2 > 0) = 0;
    edu_dis(edu_dis > 0) = 1;
end