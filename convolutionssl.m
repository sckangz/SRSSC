function [result] = convolutionssl(X,y,labelind,alpha,beta)
[~,num] = size(X');

% index of unlabeled data
ulabel = setdiff(1:num,labelind);

% initialize F
ll = length(labelind);
Fl = zeros(ll, length(unique(y)));
 
for i = 1:ll
    Fl(i, y(labelind(i))) = 1;
end
Fu = zeros(num-ll, length(unique(y)));
F=[Fl;Fu];
for j = 1:50
    % Solve S
    F_old=F;
    A = X';
    dist_f = L2_distance_1(F',F');
    % distF = sort(dist_f, 2);
    
    XX = inv((2*alpha * eye(num) + 2*(A'*A)));
    for i = 1:num
       di = dist_f(i, :);
       S(:, i) = XX*(2*A'*A(:, i) - beta*di'/2);
    end
    
    % Solve F
    S = (S+S')/2;
    D = diag(sum(S));
    L = eye(num) - D^(-1/2)*S*D^(-1/2);
    uu = zeros(num-ll, num-ll);
    for ii = 1:(num-ll)
        for jj = 1:(num-ll)
            uu(ii, jj) = L(ulabel(ii), ulabel(jj));
        end
    end
    ul = zeros(num - ll, ll);
    for ii = 1:(num - ll)
        for jj = 1:ll
            ul(ii,jj) = L(ulabel(ii), labelind(jj));
        end
    end
    Fu = -uu\(ul*Fl);
    F = [Fl; Fu];
    
    if ((j > 1)&&(norm(F - F_old, 'fro') < norm(F, 'fro') * 1e-5))
        break
    end
end
[~, uc] = size(ulabel);
[~, max_ind] = max(Fu,[],2);
cnt = 0;
for i = 1:uc
    if max_ind(i) == y(ulabel(i))
        cnt = cnt+1;
    end
end
Acc = cnt/uc;
result = Acc;