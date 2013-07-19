% Square border of figure by stretching axes LR margin to match max(UD) margins
%%
function axmarg(fi)
    figpos = get(fi,'position');

    ax = findobj('type','axes','parent',fi);
    axpos = cell2mat(get(ax,'pos'));

    set(ax,'units','pix');
    axpix = cell2mat(get(ax,'pos'));

    marg = min(min([axpix(:,2),figpos(3)-(axpix(:,4)+axpix(:,2))]));

    newpos = [repmat(marg,numel(ax),1), axpix(:,2), figpos(3)-(2*repmat(marg,numel(ax),1)) axpix(:,4)]; 
    for i = 1:size(newpos)
        set(ax(i),'pos',newpos(i,:));
    end
    set(ax,'units','norm');
end
