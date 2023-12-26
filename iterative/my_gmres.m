function [x, r_hist, x_hist] = my_gmres(A, x0, b, TOL, maxIter)

% This function MY_GMRES, applies generalized minimum residual method to
% solve Ax=b system iteratively. The algorithm is implemented through the
% class notes of IAM767(Fall 23) and the book by Yousef Saad - "Iterative
% Methods for Sparse Linear Systems"

    n = size(A,1);
    m = min(n, maxIter);
    h = sparse(zeros(m+1,m));    % Allocate the Hessenberg matrix

    r_hist = [];
    x_hist = [];

    r0 = b - A*x0; 
    beta = norm(r0);
    v = zeros(n,m+1);
    v(:,1) = r0/beta;

    for j = 1:m
        w = A*v(:,j);

        for i = 1:j
            h(i,j) = dot(w, v(:,i));
            w = w - h(i,j)*v(:,i);
        end

        h(j+1,j) = norm(w);
        if h(j+1,j) == 0
            m = j;
            fprintf('Iteration stopped at %d',m)
            break
        end
        v(:,j+1) = w / h(j+1,j);

        H = h(1:j+1, 1:j);
        e1 = double(1:j+1 == 1)';    % First basis vector in j+1 dimension

        % Apply the least squares solution to find ym
        ym = H'*H \ H'*beta*e1;
        Vm = v(:,1:j);
        x = x0+Vm*ym;
        res_norm = norm (b - A*x);

        r_hist(j) = res_norm;
        x_hist(:,j) = x;

        if res_norm < TOL
            fprintf(['GMRES converged in %d iterations with %e ' ...
                'residual\n'],j, res_norm)
            break
        end

        
    end

end