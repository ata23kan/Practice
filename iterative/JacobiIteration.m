function x = JacobiIteration(A, b, x, TOL)
% 
% function  : JacobiIteration(A,b,x,TOL)
% purpose   : Solve Ax=b with Jacobi iterative method with the given
%             tolerance value TOL and initial guess x0
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
    
    if A(i,i) == 0
        fprintf('There is a zero in the diagonal')
        return
    end
    for j = 1:N
        if i~=j
            jSum = jSum + A(i,j)*x(j);
        end
    end

    x(i) = (b(i) - jSum) / A(i,i);
end
resNorm = norm(b - A*x);
errNorm = resNorm / resNorm0;
itr = itr + 1;

end
end