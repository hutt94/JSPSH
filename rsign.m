function A = rsign(X,r,kbit)
    [n,d] = size(X); 
    if r == d
        A = sign(X);
    else
        [~,idx] = sort(X,2,'descend');
        tt = idx(:,1:r);
        A = sparse(repmat((1:n)',[r 1]),tt(:),1); A = full(A);
        if size(A,2) < kbit
            A = [A zeros(size(A,1),kbit-size(A,2))];
        end
    end
end