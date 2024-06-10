function plotTriangle(x1, x2, x3)

% function plotTriangle(x1, x2, x3)
% Plot a triangle by connecting its three vertices

% Assume the vectors are column vectors

if size(x1, 2) ~= 1
    x1 = x1';
end
if size(x2, 2) ~= 1
    x2 = x2';
end
if size(x3, 2) ~= 1
    x3 = x3';
end

x = [x1(1) x2(1) x3(1)];
y = [x1(2) x2(2) x3(2)];

% Build face array
Fx = [x1(1) x2(1) x2(1) x3(1) x3(1) x1(1)];
Fy = [x1(2) x2(2) x2(2) x3(2) x3(2) x1(2)];

oFx = reshape(Fx, 2, 3);
oFy = reshape(Fy, 2, 3);

xmax = max(max(x)); xmin = min(min(x));
ymax = max(max(y)); ymin = min(min(y));

Lx = xmax-xmin;
Ly = ymax-ymin;
xmax = xmax+.1*Lx; xmin = xmin-.1*Lx;
ymax = ymax+.1*Ly; ymin = ymin-.1*Ly;

plot(oFx, oFy, 'k-', LineWidth=1.)
% axis equal
axis([xmin xmax ymin ymax])

for k=1:numel(x)
    textString = sprintf('x_{%d}''', k);

    if k == 1
        text(x(k)-.3, y(k)-1, textString, 'FontSize', 12);

    elseif k == 2
        text(x(k)-.5, y(k)-2, textString, 'FontSize', 12);

    elseif k == 3
        text(x(k)-1, y(k)+1, textString, 'FontSize', 12);
    end
    
end

end