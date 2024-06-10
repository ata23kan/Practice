function plotAnnular(xc, yc, ri, w, savePlot)
    myring = ringAnnular("Center",[xc yc], "InnerRadius", ri, "Width",w);
    show(myring)
    hold on
%     xlim([-3 3]); ylim([-3 3])
%     xline(0, 'k'); yline(0, 'k')
    plot(xc,yc,'r*')
    if savePlot == 1
        print -dpng annular
 
    end


end