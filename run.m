load jaffe_213n_676d_10c_uni.mat
warning off
alpha = [1e-4 1e-1];
beta = [1e-4 1e-1 10];
k = 4;
rate = 0.5;

num=size(X, 1);
c=length(unique(y)); % number of class
numperc=floor(num/c); % number of data per class
labelperc=floor(rate*numperc); % number of labeled data per class

% filter X
[~, S, ~] = CAN(X', c); % c: the number of clusters. which is given by data.
S = (S+S')/2;
D = diag(sum(S));
L = eye(num) - D^(-1/2)*S*D^(-1/2);
X = (eye(num) - L/2)^k * X;

fid=fopen('test.txt','a');
fprintf(fid,'%25s','jaffe_213n_676d_10c_uni.mat');
fprintf(fid,'%12s %12.6f\r\n','rate is',rate);
fprintf(fid,'%s\t %s\t %s\t %s\t\n','alpha','beta','k','result');
for t=1:20
labelindperc = sort(randperm(numperc,labelperc)); % index of labeled data selected
labelind = []; % labelind: index of known label
for i=1:c
    labelind = [labelind labelindperc+(i-1)*numperc];
end
lres = 0;
for i = 1:length(alpha)
    for j = 1:length(beta) 
%        for h = 1:length(k)
            fprintf('params\t%.7f\t%.7f\t%d\n', alpha(i), beta(j),k);
            [result] = convolutionssl(X, y, labelind, alpha(i), beta(j));
            fprintf(fid,'%.7f\t%.7f\t%d\t%.6f\t', alpha(i), beta(j), k, result);
            fprintf(fid,'\n');
            if (result > lres)
               lres = result;
            end
%        end
     end
end
acc(t) = lres;
end

 mean(acc)
 std(acc)
 fprintf(fid,'%12s %12.6f\r\n','mean',mean(acc) );
 fprintf(fid,'%12s %12.6f\r\n','std',std(acc) );
 fclose(fid);