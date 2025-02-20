clear all;
close all;
clc

% bump_method = 'Gaussian';
bump_method = 'Cosine';

switch bump_method
  case 'Gaussian'
    Gaussian_bump();
  case 'Cosine'
    Cosine_bump();
end

function Gaussian_bump()
  % Gaussian Bump Animation
  % Parameters
  A = 0.2;            % Bump height
  sigma = 0.5;      % Standard deviation (controls width)
  W = 1.0;            % Advection speed
  x = linspace(-5,5,500);  % Spatial domain
  x0 = -2;
  tmax = 10;        % Maximum time
  dt = 0.05;        % Time step for animation
  
  figure;
  for t = 0:dt:tmax
      % Define Gaussian bump: centered at x = xc
      xc = x0 + W*t;
      y = A * exp(-((x - xc).^2)/(2*sigma^2));
      
      % Plot the bump
      plot(x, y, 'r-', 'LineWidth', 2);
      title(sprintf('Gaussian Bump at time t = %.2f', t));
      xlabel('x');
      ylabel('f(x,t)');
      axis([-5 5 -0.1 A+0.1]);
      grid on;
      drawnow;
      pause(0.05); % Pause for animation effect
  end
end

function Cosine_bump()
  % Cosine Bump Animation
  % Parameters
  A = 1;            % Bump height
  L = 2;            % Bump length (total width)
  W = 1;            % Advection speed
  x = linspace(0,5,500);  % Spatial domain
  tmax = 10;        % Maximum time
  dt = 0.05;        % Time step for animation
  
  figure;
  for t = 0:dt:tmax
      % Initialize bump values
      y = zeros(size(x));
      
      % Compute indices where the bump is defined
      idx = abs(x - W*t) <= L/2;
      
      % Define cosine bump: maximum at x = W*t and zero at the edges
      y(idx) = A * cos(pi*(x(idx) - W*t)/L);
      
      % Plot the bump
      plot(x, y, 'b-', 'LineWidth', 2);
      title(sprintf('Cosine Bump at time t = %.2f', t));
      xlabel('x');
      ylabel('f(x,t)');
      axis([0 5 -0.1 A+0.1]);
      grid on;
      drawnow;
      pause(0.05); % Pause for animation effect
  end
end

