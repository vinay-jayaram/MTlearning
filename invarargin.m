% Function to parse varargin
% Returns [] if string 'option' is not in varargin (var) and value assigned to 'option' otherwise
% 09/11/2009 by Moritz Grosse-Wentrup (moritzgw@ieee.org)
%
% Call: out = invarargin(var,option)
%

function out = invarargin(var,option)

if isempty(find(strcmp(var,option)))
    out = [];
else
    out = var{find(strcmp(var,option))+1};
end
