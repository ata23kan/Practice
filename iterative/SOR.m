function x = SOR(A, b, x, w, TOL)
% 
% function  : SOR(A,b,x,TOL)
% purpose   : Solve Ax=b with Gauss-Seidel iterative method with the given
%             tolerance value TOL and initial guess x
% 
N = size(A,1);
r0 = b - A*x;  % Initial residual

% Initialize the residuals and tolerance criteria
resNorm0 = norm(r0);
resNorm  = Inf;
errNorm = resNorm/resNorm0;

itr = 0;

while errNorm > TOL
for i = 1:N
    jSum = 0 ;      % Initialize the sum in the right hand side
    xold = x;       % Store the previous level solution 
    
    if A(i,i) == 0
        % Check if there is a zero on the diagonal
        fprintf('There is a zero in the diagonal')
        return
    end

    for j = 1:i-1
        jSum = jSum + A(i,j)*x(j);
    end

    for j = i+1:N
        jSum = jSum + A(i,j)*xold(j);
    end
    
    x(i) = (1-w)*xold(i) + w*(b(i) - jSum) / A(i,i);
end

resNorm = norm(b - A*x);
errNorm = resNorm / resNorm0;
itr = itr + 1;

end
end