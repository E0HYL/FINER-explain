import numpy as np
import cvxpy as cvx
from sklearn import linear_model
import multiprocessing
import time
from functools import partial

from lime.lime_base import LimeBase
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer


def r2_score(pred, label):
    # how well the predictor variables can explain the variation in the response variable
    return 1 - ((pred - label)**2).sum()/((label - label.mean())**2).sum()


class LemnaBase(LimeBase):
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None, 
                 run_lime=False):
        super().__init__(kernel_fn, verbose, random_state)
        self.run_lime = run_lime

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, distances, label, num_features, feature_selection='none', model_regressor=None): # run_lime: True (do not need to perturbate the data for a second time).
        neighborhood_labels = neighborhood_labels[:, label]
        beta = em_regression_algorithm(neighborhood_data, neighborhood_labels) # , K, alpha_S, iterations, linreg_type

        if beta is not None:
            predictions = neighborhood_data @ beta
            prediction_score = r2_score(predictions, neighborhood_labels)
            local_pred = predictions[:1] # the first prediction: the sample to be explained

            lemna_explanation = (.0, # intercept_: Independent term in decision function
                list(zip( range(len(beta)), beta)), # explanations: list of tuples (x -> feature id, y -> local weight)
                prediction_score, # regression score: cofficient of determination R^2
                local_pred, # the prediction of the explanation model on the original instance
            )
        else:
            lemna_explanation = (None, None, None, None)

        if self.run_lime:
            lime_explanation = super().explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features, feature_selection, model_regressor)
            return lemna_explanation, lime_explanation

        return lemna_explanation # lime_explanation


def check_kernel_fn(kernel_width, kernel, num_features=None):
    if kernel_width is None:
        kernel_width = np.sqrt(num_features) * .75
    kernel_width = float(kernel_width)

    if kernel is None:
        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    kernel_fn = partial(kernel, kernel_width=kernel_width)
    return kernel_fn


class LemnaImageExplainer(LimeImageExplainer):
    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='none', random_state=None, run_lime=False):
        super().__init__(kernel_width, kernel, verbose, feature_selection, random_state)
        kernel_fn = check_kernel_fn(kernel_width, kernel)

        self.base = LemnaBase(kernel_fn, verbose, random_state=self.random_state, run_lime=run_lime)


class LemnaTabularExplainer(LimeTabularExplainer):
    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='none',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None,
                 run_lime=False):
        super(LemnaTabularExplainer, self).__init__(training_data, mode, training_labels, feature_names, categorical_features, categorical_names, kernel_width, kernel, verbose, class_names, feature_selection, discretize_continuous, discretizer, sample_around_instance, random_state, training_data_stats)
        kernel_fn = check_kernel_fn(kernel_width, kernel, num_features=training_data.shape[1])
        self.base = LemnaBase(kernel_fn, verbose, random_state=self.random_state, run_lime=run_lime)


# gaussian density function
def gaussian(x, mu, sigma_squared):
    eps = 0
    return 1/(np.sqrt(2*np.pi*sigma_squared)+eps)*np.exp(-0.5*(x-mu)**2/(sigma_squared+eps))


# the expectation maximization algorithm of the lemna paper, calculation of indices can be found in appendix
# number of iterations: https://github.com/Henrygwb/Explaining-DL/blob/b443f7f61a8e860db88551dbb27a62d2f26a14fa/dmm-mem/code/dmm_men.r#L63
def em_regression_algorithm(data, labels, K=3, alpha_S=1e-3, iterations=20000, linreg_type='fused_lasso', verbose=False, save_path=None):
    # determine if data is sparse
    sparse = True if type(data).__module__ == 'scipy.sparse.csr' else False
    no_samples = data.shape[0]
    sample_len = data.shape[1]
    label_sum = np.sum(np.abs(labels))
    no_ones = len(np.where(labels == 1)[0])
    no_zeros = len(labels) - no_ones
    #data = (data-np.mean(data))/np.std(data)
    if linreg_type not in ['lasso', 'fused_lasso']:
        print('Invalid linreg_type (%s)' %linreg_type)
        exit(1)
    elif sample_len <=1:
        if verbose:
            print('Encountered invalid data sample!')
        with open(save_path, 'a') as f:
            print('Invalid sample.', file=f)
        return np.array([-1]), np.array([-1])
    eps = 1e-6
    number_of_history_betas = 3
    convergence_threshold = 1e-2
    # initialize the parameters randomly
    pi, sigma_sq = np.random.uniform(0, 1, size=K), np.random.uniform(0, 1, size=K)
    # normalize pi
    pi = 1/np.sum(pi) * pi
    beta = np.random.uniform(-.1, .1, size=(K, sample_len))
    z_hat = np.zeros(shape=(no_samples, K))
    # check for convergence using last betas
    old_likelihoods = []
    converged = False
    # run at most 'iterations' iterations but finish if the last 'number of history betas' log likelihood values are
    # close to each other
    initial_log_likelihood = 0
    for n in range(no_samples):
        if sparse:
            likelihood = sum([pi[k] * gaussian(labels[n], data.getrow(n).dot(beta[k,:])[0], sigma_sq[k])
                              for k in range(K)])
        else:
            likelihood = sum([pi[k] * gaussian(labels[n], np.dot(data[n,:],beta[k, :]), sigma_sq[k])
                              for k in range(K)])
        if likelihood != 0:
            initial_log_likelihood += np.log(likelihood)
    if verbose:
        print('Starting Expectation maximization algorithm for %d iterations with sum of labels %d' %(iterations,
                                                                                                      label_sum))
    start_time = time.time()
    for iter in range(iterations):
        # E step
        for i in range(no_samples):
            if sparse:
                denom_e = sum([pi[k] * gaussian(labels[i], data.getrow(i).dot(beta[k, :])[0], sigma_sq[k]) for k in
                             range(K)])
            else:
                denom_e = sum([pi[k] * gaussian(labels[i], np.dot(data[i,:], beta[k,:]), sigma_sq[k]) for k in
                             range(K)])
            if denom_e == 0:
                denom_e = eps
                if verbose:
                    print('set denom_e to eps')
            for k in range(K):
                pred_2 = data.getrow(i).dot(beta[k, :])[0] if sparse else np.dot(data[i,:], beta[k,:])
                z_hat[i, k] = pi[k]*gaussian(labels[i], pred_2, sigma_sq[k])/denom_e
        # M step
        for k in range(K):
            denom_m = np.sum(z_hat[:, k])
            if denom_m == 0:
                denom_m = eps
            if sparse:
                sigma_sq[k] = sum([z_hat[i, k] * (labels[i] - data.getrow(i).dot(beta[k, :])[0])**2 for i in
                                   range(no_samples)]) / denom_m
            else:
                sigma_sq[k] = sum([z_hat[i, k] * (labels[i] - np.dot(data[i, :], beta[k, :])) ** 2 for i in
                                   range(no_samples)]) / denom_m
            if sigma_sq[k] == 0:
                sigma_sq[k] += eps
                if verbose:
                    print('added eps to sigma')
            pi[k] = np.sum(z_hat[:, k])/no_samples
        component_assignments = np.argmax(z_hat, axis=1)
        # estimate betas by linear regression with fused lasso loss
        for k in range(K):
            sample_indices_of_k = np.where(component_assignments == k)[0]
            samples_of_k = data[sample_indices_of_k, :]
            labels_of_k = labels[sample_indices_of_k]
            if len(labels_of_k) > 0:
                if linreg_type == 'fused_lasso':
                    try:
                        beta[k,:] = solve_fused_lasso_regression(samples_of_k, labels_of_k, alpha_S)
                    except cvx.error.SolverError:
                        return None
                elif linreg_type == 'lasso':
                    reg = linear_model.Lasso(alpha=alpha_S, precompute=True, normalize=True, max_iter=3000)
                    reg.fit(samples_of_k, labels_of_k)
                    beta[k, :] = reg.coef_
        # recompute log_likelihood in order to check for convergence
        log_likelihood = 0
        for n in range(no_samples):
            if sparse:
                likelihood = sum([pi[k] * gaussian(labels[n], data.getrow(n).dot(beta[k, :])[0], sigma_sq[k])
                                  for k in range(K)])
            else:
                likelihood = sum([pi[k] * gaussian(labels[n], np.dot(data[n, :], beta[k, :]), sigma_sq[k])
                                  for k in range(K)])
            if likelihood != 0:
                log_likelihood += np.log(likelihood)
        if len(old_likelihoods) < number_of_history_betas:
            old_likelihoods.append(log_likelihood)
        else:
            abs_diffs = []
            for beta_idx in range(number_of_history_betas):
                diff = np.abs(old_likelihoods[beta_idx]-log_likelihood)
                abs_diffs.append(diff)
            convergence_check = [np.sum(diff <= convergence_threshold) for diff in abs_diffs]
            if np.sum(convergence_check) == number_of_history_betas:
                converged = True
            old_likelihoods.pop(0)
            old_likelihoods.append(log_likelihood)
        if verbose:
            print('likelihood history', old_likelihoods)
        if converged:
            end_time = time.time()
            if verbose:
                print('EM-Alogirthm converged after %d iterations (%d seconds).' %(iter, end_time-start_time))
                argm = np.argmax(z_hat, axis=1)
                for k in range(K):
                    indices_of_k = np.where(argm==k)[0]
                    labels_of_k = labels[indices_of_k]
                    labels_in_k = np.unique(labels_of_k)
                    d = {}
                    for label in labels_in_k:
                        d[label] = len(np.where(labels_of_k==label)[0])
                    print('labels in cluster %d'%k, d)
            break
    if save_path:
        with open(save_path, 'a') as f:
            projections = np.dot((beta * pi[:, np.newaxis]), np.transpose(data))
            projections = np.sum(projections, axis=0)
            diff = (projections - labels) ** 2
            mse = 1. / len(diff) * np.sum(diff)
            if converged:
                print('S=%.3f_K=%d_linreg_type=%s_no_ones=%d_no_zeros=%d_time=%.4f_mse=%.4f'
                      % (alpha_S, K, linreg_type, no_ones, no_zeros, end_time - start_time, mse), file=f)
            else:
                print('S=%.3f_K=%d_linreg_type=%s_no_ones=%d_no_zeros=%d_time=%.4f_mse=%.4f'
                      % (alpha_S, K, linreg_type, no_ones, no_zeros, -1, mse), file=f)
            # print('mse', mse)
    # return the parameters by choosing the cluster belonging to the first row of the perturbations which is by
    # assumption the sample to be explained
    cluster_idx_sample = np.argmax(z_hat[0])
    return beta[cluster_idx_sample] #, sigma_sq[cluster_idx_sample]


# returns matrix A such that sum(abs(A*x)) is the fused lasso constraint on x
def get_band_matrix_fused_lasso(dim):
    if dim <= 1:
        print('Invalid dimension for band matrix (%d)!'%dim)
        return None
    A = np.diag(-1*np.ones(dim))
    rng = np.arange(dim-1)
    A[rng, rng+1] = 1
    A[dim-1,:] = 0
    return A


def solve_fused_lasso_regression(samples, labels, S, positive=False):
    # for the sake of clarity
    A = cvx.Constant(samples)
    no_dimensions = samples.shape[1]
    beta = cvx.Variable(no_dimensions)
    # careful: the band matrix can get large very fast if dimension is high
    # D = get_band_matrix_fused_lasso(no_dimensions)
    regularization = beta[1:] - beta[:no_dimensions - 1]
    objective = cvx.Minimize(cvx.sum_squares(A@beta - labels)) # @ for matrix-matrix and matrix-vector multiplication
    # the constraint is the sum of the (absolute) differences of the neighbored betas to be bounded by S
    # constraints = [cvx.sum(cvx.abs(D*beta)) <= S]
    constraints = [cvx.sum(cvx.abs(regularization)) <= S]
    problem = cvx.Problem(objective, constraints)
    if positive: # try another solver if failed
        try:
            problem.solve()
        except cvx.error.SolverError:
            problem.solve(solver=cvx.ECOS)
    else:
        problem.solve()
    return beta.value


def lemna_parallel(perturbation_data, perturbation_labels, K=3, alpha_S=1e-3, iterations=5, no_processes=2, linreg_type='fused_lasso',
                   repetitions=1, verbose=False, save_path=None):
    assert len(perturbation_data) == len(perturbation_labels)
    no_samples = len(perturbation_data) * repetitions
    # repeat each perturbation repetitions times for parallel processing
    perturbations_repeated = []
    for p in perturbation_data:
        perturbations_repeated += [p]*repetitions
    labels_repeated = np.repeat(perturbation_labels, repetitions, axis=0)
    if save_path:
        filenames = np.array([save_path+str(i) for i in range(len(perturbation_data))])
        filenames = np.repeat(filenames, repetitions)
    else:
        filenames = no_samples*[None]
    arg_gen = zip(perturbations_repeated, labels_repeated, no_samples*[K], no_samples*[alpha_S], no_samples*[iterations],
                  no_samples*[linreg_type], no_samples*[verbose], filenames)
    with multiprocessing.Pool(processes=no_processes) as pool:
        lemna_betas = pool.starmap(em_regression_algorithm, arg_gen)
    if type(perturbation_data) is list:
        betas = [lemna_beta[0] for lemna_beta in lemna_betas]
    else:
        betas = np.array([lemna_beta[0] for lemna_beta in lemna_betas]).reshape((len(perturbation_data), repetitions,
                                                                                 perturbation_data.shape[-1]))
    sigmas = np.array([lemna_beta[1] for lemna_beta in lemna_betas]).reshape((len(perturbation_data), repetitions))
    return betas, sigmas

