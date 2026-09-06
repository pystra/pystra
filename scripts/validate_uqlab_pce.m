function validate_uqlab_pce(uqlab_root, fixture_path)
% Regenerate numerical references with unmodified UQLab 2.2.0 routines.
% Works with MATLAB or Octave; no UQLab framework initialization is needed.
% Inputs and uniform bootstrap indices are stored in the JSON fixture.
% The exhaustive degree/q loops compare UQLab's final hybrid-OLS LOO score.
addpath(genpath(uqlab_root));
data = jsondecode(fileread(fixture_path));
for c = 1:numel(data.cases)
    current = data.cases(c);
    points = current.points;
    values = current.values(:);
    best_error = Inf;
    options.early_stop = false;
    options.normalize = true;
    options.hybrid_lars = true;
    options.loo_modified = true;
    options.loo_hybrid = true;
    options.display = 0;
    for degree = current.degrees(:)'
        for q = current.q_norms(:)'
            trunc.qNorm = q;
            indices = full(uq_generate_basis_Apmj(0:degree,size(points,2),trunc));
            basis = evaluate_basis(points, indices);
            fitted = uq_lar(basis,values,options);
            if fitted.LOO < best_error
                best_error = fitted.LOO;
                selected = fitted.nz_idx;
                chosen_indices = indices(selected,:);
                coefficients = fitted.coefficients(selected);
                expected.degree = degree;
                expected.q_norm = q;
                expected.indices = chosen_indices;
                expected.coefficients = coefficients;
                expected.loo_error = fitted.optErrorParams.loo;
                expected.corrected_loo_error = fitted.LOO;
            end
        end
    end
    query_basis = evaluate_basis(current.query,chosen_indices);
    expected.mean = query_basis*coefficients;
    training_basis = evaluate_basis(points,chosen_indices);
    boot_predictions = zeros(size(current.query,1),size(current.bootstrap_indices,1));
    for b = 1:size(current.bootstrap_indices,1)
        rows = current.bootstrap_indices(b,:)+1;
        fitted = uq_PCE_OLS_regression(training_basis(rows,:), values(rows));
        boot_predictions(:,b) = query_basis*fitted.coefficients;
    end
    expected.std = std(boot_predictions,0,2);
    data.cases(c).expected = expected;
end
file = fopen(fixture_path,'w');
fwrite(file,jsonencode(data));
fclose(file);
end

function basis = evaluate_basis(points,indices)
basis = ones(size(points,1),size(indices,1));
for axis = 1:size(points,2)
    values = uq_eval_hermite(max(indices(:,axis)),points(:,axis));
    basis = basis.*values(:,indices(:,axis)+1);
end
end
