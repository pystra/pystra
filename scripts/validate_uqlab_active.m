function validate_uqlab_active(uqlab_root, fixture_path)
% Compare fixed-trend/fixed-kernel universal prediction and FBR with the
% unmodified UQLab 2.2.0 numerical routines; no framework initialization.
% Kernel and Hermite adapters below specify the fixture's mathematical model.
addpath(genpath(uqlab_root));
data = jsondecode(fileread(fixture_path));
for c = 1:numel(data.cases)
    current = data.cases(c);
    model.Internal.Runtime.M = size(current.points,2);
    model.Internal.Runtime.Nout = 1;
    model.Internal.Runtime.nonConstIdx = 1:size(current.points,2);
    model.Internal.Runtime.isCalculated = true;
    model.Internal.Runtime.isCustom = true;
    model.Internal.Scaling = false;
    model.Internal.KeepCache = true;
    model.Internal.Powers = current.indices;
    model.ExpDesign.U = current.points;
    model.ExpDesign.Y = current.values(:);
    F = evaluate_trend(current.points,model);
    corr_options.Nugget = current.noise;
    R = evaluate_correlation(current.points,current.points,current.length_scale,corr_options);
    aux = uq_Kriging_calc_auxMatrices(R,F,current.values(:),'default');
    beta = aux.FTRinvF \ (aux.FTRinv*current.values(:));
    residual = current.values(:)-F*beta;
    variance = (residual'*(R\residual))/numel(residual);
    model.Internal.Kriging.Trend.F = F;
    model.Internal.Kriging.Trend.Handle = @evaluate_trend;
    model.Internal.Kriging.Trend.Type = 'custom';
    model.Internal.Kriging.Trend.beta = beta;
    model.Internal.Kriging.Optim.Theta = current.length_scale;
    model.Internal.Kriging.GP.Corr = corr_options;
    model.Internal.Kriging.GP.Corr.Handle = @evaluate_correlation;
    model.Internal.Kriging.GP.R = R;
    model.Internal.Kriging.GP.sigmaSQ = variance;
    model.Internal.Kriging.Cached = aux;
    [expected.mean,expected.variance] = uq_Kriging_eval(model,current.query);
    expected.coefficients = beta;
    expected.process_variance = variance;
    data.cases(c).expected = expected;
end
% UQLab's RBDO selector maximizes the negative agreement. PySTRA minimizes
% positive agreement, so only the sign of the score convention differs.
data.fbr.expected = -uq_LF_FBR(data.fbr.predictions);
file = fopen(fixture_path,'w');
fwrite(file,jsonencode(data));
fclose(file);
end

function basis = evaluate_trend(points,model)
powers = model.Internal.Powers;
basis = ones(size(points,1),size(powers,1));
for axis = 1:size(points,2)
    values = uq_eval_hermite(max(powers(:,axis)),points(:,axis));
    basis = basis.*values(:,powers(:,axis)+1);
end
end

function correlation = evaluate_correlation(first,second,scales,options)
correlation = zeros(size(first,1),size(second,1));
for i = 1:size(first,1)
    differences = bsxfun(@rdivide,bsxfun(@minus,second,first(i,:)),scales(:)');
    correlation(i,:) = exp(-0.5*sum(differences.^2,2))';
end
if options.Nugget > 0
    correlation = correlation + options.Nugget*eye(size(first,1));
end
end
