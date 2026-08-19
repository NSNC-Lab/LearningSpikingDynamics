function DrawBox( x,y )

hold on;
plot([x+0.5,x+0.5],[y-0.5,y+0.5],'k-');
hold on;
plot([x-0.5,x-0.5],[y-0.5,y+0.5],'k-');
hold on;
plot([x-0.5,x+0.5],[y+0.5,y+0.5],'k-');
hold on;
plot([x-0.5,x+0.5],[y-0.5,y-0.5],'k-');
hold off;


end

