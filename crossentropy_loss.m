function loss = crossentropy_loss(pred_y, true_y, penalty)
%CROSS_ENTROPY_LOSS loss function for the cross-entropy error.
    ce1 = true_y .* log(pred_y+1e-100);
    ce2 = (1-true_y) .* log(1-pred_y+1e-100);
    loss = -sum(ce1 + ce2);
    if exist('penalty', 'var')
        loss = loss + penalty;
    end
end

