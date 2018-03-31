function []=save2mat(C,mode,name,n,varargin)
varname = strcat('iter_',name);
if(exist('./result','dir')==0)
    mkdir('result')
end
matname = sprintf('./result/%s%sn%d',mode,name,n);
if(~isempty(varargin))
    switch varargin{1}
        case 'flat'
            matname= strcat(matname,'flat');
            varname = strcat(varname,'flat');
        case 'normal'
            if(length(varargin)==3)
                if(varargin{3}<1)
                    tmp = varargin{3}*10;
                    matname = sprintf('%s%s%d0_%d',matname,varargin{1},varargin{2},tmp);
                    varname = sprintf('%s%s%d0_%d',varname,varargin{1},varargin{2},tmp);
                else
                    matname = sprintf('%s%s%d%d',matname,varargin{1},varargin{2},varargin{3});
                    varname = sprintf('%s%s%d%d',varname,varargin{1},varargin{2},varargin{3});                    
                end

            else
                error('Specify mean and variance for normal prior');
            end
        otherwise
            error('Incompatible Prior')
    end
end
matname = strcat(matname,'.mat');
if(exist(matname,'file')==2)
    load(matname)
else
    eval(sprintf('%s = [];', varname));
    save(matname,varname)
end
eval(sprintf('%s = [%s C];', varname, varname));
save(matname,varname)
end