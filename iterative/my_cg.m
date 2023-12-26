function [x, r_hist, x_hist] = my_cg(A, x0, b, TOL)
% This function MY_CG, applies generalized minimum residual method to
% solve Ax=b system iteratively. The algorithm is implemented through the
% class notes of IAM767(Fall 23) and the book by Yousef Saad - "Iterative
% Methods for Sparse Linear Systems"    

    x = x0;
    r = b-A*x;
    p = r;
    r_hist = [];
    x_hist = [];

    j = 1;

    while norm(r) > TOL
        r_old = r;
        
        Ap = A*p;
        alpha = dot(r,r) / dot(Ap,p);
        x = x + alpha*p;
        r = r-alpha * Ap;
        beta = dot(r,r) / dot(r_old, r_old);
        p = r + beta*p;

        r_hist(j) = norm(r);
        x_hist(:,j) = x;
        j = j+1;

    end
fprintf(['CG converged in %d iterations with %e ' ...
                'residual\n'],j, norm(r))

end