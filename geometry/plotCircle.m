function [xunit, yunit] = plotCircle(x,y,r,savePlot)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
plot(xunit, yunit, 'k', 'LineWidth',2);
plot(x,y,'r*')
grid on
axis square
xlim([-7 7]); ylim([-7 7])
xline(0, 'k'); yline(0, 'k')
hold off

if savePlot == 1
    print -dpng circle_r2
 
end
