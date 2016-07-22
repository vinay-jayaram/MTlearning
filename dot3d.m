function Ab = dot3d(A, b)
%MULTI_DOT Computes A*b along the first dimension of the 3D tensor A and
% vector b. Hence, a matrix multiplication of the last two dimensions
% with b is performed.
    dim = size(A);
    Ab = zeros(dim(1), dim(2));
    for i = 1:dim(1)
        Ab(i, :) = squeeze(A(i, :, :))*b;
    end
end

