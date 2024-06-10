% Torus

[theta,phi] = meshgrid(linspace(0,2*pi,50));
r = 1;
R = 2.0;
x = (R + r*cos(theta)).*cos(phi);
y = (R + r*cos(theta)).*sin(phi);
z = r*sin(theta);
surf(x,y,z)
shading interp
colormap hot
axis equal