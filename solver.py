import numpy as np
import cvxpy as cp


class Solver():
    def __init__(self, loss_func, solver_name):
        assert loss_func in ['lr', 'huber']
        self.loss_func = loss_func
        if loss_func == 'lr':
            self.c = 0.25
        elif loss_func == 'huber':
            self.c = 1.0

        solver_dict = {
            'no_privacy': self.solve_no_privacy,
            'output_pert': self.solve_output_pert,
            'obj_pert': self.solve_obj_pert,
            'no_privacy_ew': self.solve_no_privacy_ew,
            'output_pert_ew': self.solve_output_pert_ew,
            'obj_pert_ew': self.solve_obj_pert_ew,
            'no_privacy_gw': self.solve_no_privacy_gw,
            'output_pert_gw': self.solve_output_pert_gw,
            'obj_pert_gw': self.solve_obj_pert_gw,
            'dp_sgd': self.solve_dp_sgd,
            'dp_sgd_gw': self.solve_dp_sgd_gw}
        assert solver_name in solver_dict.keys()
        self.solver = solver_dict[solver_name]

    def generate_noise(self, beta, dim):
        '''
        norm of b from Gamma(d, 1/beta) and direction of b uniformly
        b \in R^{dim}
        '''
        norm = np.random.gamma(dim, 1./beta)
        d = np.random.uniform(size=dim)
        return norm * d / np.linalg.norm(d)

    def calc_sample_weight(self, Y, weight=None):
        if weight is None:
            weight = ((Y == 1) * sum(Y == -1) / len(Y)
                      + (Y == -1) * sum(Y == 1) / len(Y))
            return weight
        else:
            return (Y == 1) * (1 - weight) + (Y == -1) * weight

    def _solve(self, problem):
        try:
            problem.solve(solver=cp.SCS)
            return True
        except cp.SolverError:
            pass
        try:
            problem.solve()
            return True
        except cp.SolverError:
            return False

    def solve(self, X, Y, eps, *para):
        return self.solver(X, Y, eps, *para)

    def margin(self, X, Y, w):
        return cp.multiply(Y, X @ w)

    def loss(self, X, Y, w):
        z = self.margin(X, Y, w)
        if self.loss_func == 'lr':
            return cp.logistic(-z)
        elif self.loss_func == 'huber':
            h = 0.5
            return cp.multiply(1/(4*h), cp.huber(cp.minimum(0, z-1-h), 2*h))

    def solve_no_privacy(self, X, Y, eps, lambd_val):
        n, m = X.shape[0], X.shape[1]
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        log_likelihood = (cp.sum(self.loss(X, Y, w)) / n
                          + lambd * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_no_privacy_ew(self, X, Y, eps, lambd_val):
        n, m = X.shape[0], X.shape[1]
        sw = self.calc_sample_weight(Y)
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        log_likelihood = (cp.sum(cp.multiply(sw,
                          self.loss(X, Y, w))) / n
                          + lambd * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_no_privacy_gw(self, X, Y, eps, lambd_val, gw=0.5):
        n, m = X.shape[0], X.shape[1]
        assert 0 <= gw <= 1
        sw = self.calc_sample_weight(Y, gw)
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        log_likelihood = (cp.sum(cp.multiply(sw,
                          self.loss(X, Y, w))) / n
                          + lambd * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_output_pert(self, X, Y, eps, lambd_val):
        '''
        Algorithm 1
        '''
        n, m = X.shape[0], X.shape[1]
        w = self.solve_no_privacy(X, Y, eps, lambd_val)
        if w is not None:
            b = self.generate_noise(n * lambd_val * eps / 2, m)
            return w + b[:, None]
        else:
            return None

    def solve_output_pert_ew(self, X, Y, eps, lambd_val):
        '''
        Algorithm 1 with sample weight
        '''
        n, m = X.shape[0], X.shape[1]
        w = self.solve_no_privacy_ew(X, Y, eps, lambd_val)
        if w is not None:
            beta = n * lambd_val * eps / 3
            b = self.generate_noise(beta, m)
            return w + b[:, None]
        else:
            return None

    def solve_output_pert_gw(self, X, Y, eps, lambd_val, gw):
        '''
        Algorithm 1 with sample weight
        '''
        n, m = X.shape[0], X.shape[1]
        assert 0 <= gw <= 1
        w = self.solve_no_privacy_gw(X, Y, eps, lambd_val, gw)
        if w is not None:
            beta = n * lambd_val * eps / (2 * (1 - gw))
            b = self.generate_noise(beta, m)
            return w + b[:, None]
        else:
            return None

    def solve_obj_pert(self, X, Y, eps, lambd_val):
        '''
        Algorithm 2
        c = 1/4 for logistic regression
        c = 1 for huber svm
        '''
        n, m = X.shape[0], X.shape[1]
        eps_prime = eps - 2 * np.log(1 + self.c/(n*lambd_val))
        if eps_prime > 0:
            delta = 0
        else:
            eps_prime = eps / 2
            delta = self.c/(n*(np.exp(eps/4)-1)) - lambd_val
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        b = self.generate_noise(eps_prime/2, m)
        log_likelihood = (cp.sum(self.loss(X, Y, w)) / n
                          + lambd * cp.norm(w, 2)**2 / 2
                          + b @ w / n
                          + delta * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_obj_pert_ew(self, X, Y, eps, lambd_val):
        '''
        Algorithm 2 with sample weight
        c = 1/4 for logistic regression
        c = 1 for huber svm
        '''
        n, m = X.shape[0], X.shape[1]
        sw = self.calc_sample_weight(Y)
        eps_prime = (eps - (2 * np.log(1 + self.c/(n*lambd_val)) + (n-1)
                     * np.log(1 + self.c/(n*n*lambd_val))))
        if eps_prime > 0:
            delta = 0
        else:
            delta = self.c/(n*(np.exp(eps/(2*(n+1)))-1)) - lambd_val
            eps_prime = (eps * n / (n+1) - (n-1)
                         * np.log(1 + (np.exp(eps/(2*(n+1))) - 1) / n))
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        b = self.generate_noise(eps_prime/3, m)
        log_likelihood = (cp.sum(cp.multiply(sw,
                          self.loss(X, Y, w))) / n
                          + lambd * cp.norm(w, 2)**2 / 2
                          + b @ w / n
                          + delta * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_obj_pert_gw(self, X, Y, eps, lambd_val, gw):
        '''
        Algorithm 2 with sample weight
        c = 1/4 for logistic regression
        c = 1 for huber svm
        '''
        n, m = X.shape[0], X.shape[1]
        assert 0 <= gw <= 1
        sw = self.calc_sample_weight(Y, gw)
        eps_prime = eps - (2 * np.log(1 + self.c*(1-gw)/(n*lambd_val)))
        if eps_prime > 0:
            delta = 0
        else:
            delta = self.c*(1-gw)/(n*(np.exp(eps/4)-1)) - lambd_val
            eps_prime = eps / 2
        w = cp.Variable((m, 1))
        lambd = cp.Parameter(nonneg=True, value=lambd_val)
        b = self.generate_noise(eps_prime/(2*(1-gw)), m)
        log_likelihood = (cp.sum(cp.multiply(sw,
                          self.loss(X, Y, w))) / n
                          + lambd * cp.norm(w, 2)**2 / 2
                          + b @ w / n
                          + delta * cp.norm(w, 2)**2 / 2)
        problem = cp.Problem(cp.Minimize(log_likelihood))
        return w.value if self._solve(problem) else None

    def solve_dp_sgd_base(self, X, Y, eps, lambd_val, mb, k, gw=None):
        '''
        only dealt with the strongly convex case for comparison
        '''
        n, m = X.shape[0], X.shape[1]
        w = np.zeros((m, 1))
        R = 1.0 / lambd_val
        t = 0
        for _ in range(k):
            random_indices = np.random.permutation(n)
            for i in range(n // mb):
                t += 1
                eta = self.step_size(t, lambd_val, gw)
                interval = range(i*mb, (i+1)*mb)
                indices = random_indices[interval]
                xs, ys = X[indices], Y[indices]
                g = self.grad(xs, ys, w, lambd_val, gw)
                w = w - (eta / mb) * g
                w = self.project(w, R)
        sens = self.sensitivity(lambd_val, R, n, gw)
        b = self.generate_noise(eps/sens, m)
        return w + b[:, None]

    def solve_dp_sgd(self, X, Y, eps, lambd_val, mb, k):
        return self.solve_dp_sgd_base(X, Y, eps, lambd_val, mb, k)

    def solve_dp_sgd_gw(self, X, Y, eps, lambd_val, mb, k, gw):
        return self.solve_dp_sgd_base(X, Y, eps, lambd_val, mb, k, gw)

    def grad(self, X, Y, w, lambd_val, gw=None):
        n = len(Y)
        zp = X @ w
        zm = -X @ w
        if self.loss_func == 'lr':
            lpp = -1 / (1 + np.exp(zp))
            lpm = -1 / (1 + np.exp(zm))
        elif self.loss_func == 'huber':
            h = 0.5
            lpp = ((zp > 1 + h) * 0
                   + (np.abs(1-zp) <= h) * (-1.0 * (1 + h - zp) / (2 * h))
                   + (zp < 1 - h) * -1)
            lpm = ((zm > 1 + h) * 0
                   + (np.abs(1-zm) <= h) * (-1.0 * (1 + h - zm) / (2 * h))
                   + (zm < 1 - h) * -1)

        if gw is None:
            lgrad = np.sum(((Y == 1) * lpp - (Y == -1) * lpm) * X,
                           axis=0)[:, None]
        else:
            lgrad = np.sum(((1-gw) * (Y == 1) * lpp
                            - gw * (Y == -1) * lpm) * X, axis=0)[:, None]

        ngrad = n * lambd_val * w
        return lgrad + ngrad

    def project(self, w, R):
        '''
        project w in the hypercube with radius R
        '''
        norm = np.linalg.norm(w, ord=2)
        if norm <= R:
            return w
        else:
            return (R / norm) * w

    def sensitivity(self, lambd_val, R, n, gw=None):
        if self.loss_func == 'lr':
            L = 1.0 if gw is None else 1-gw
            L += lambd_val * R
            gamma = lambd_val
        elif self.loss_func == 'huber':
            L = 1.0 if gw is None else 1-gw
            L += lambd_val * R
            gamma = lambd_val
        return (2 * L) / (gamma * n)

    def step_size(self, t, lambd_val, gw=None):
        if self.loss_func == 'lr':
            # beta for logistc regression is 1/4 + lambda
            # but follow the description of the paper
            beta = 1.0 if gw is None else 1-gw
            beta += lambd_val
            gamma = lambd_val
        elif self.loss_func == 'huber':
            h = 0.5
            beta = 1.0 / (2 * h) if gw is None else (1-gw) / (2 * h)
            beta += lambd_val
            gamma = lambd_val
        return min(1.0 / beta, 1.0 / (gamma * t))


if __name__ == '__main__':
    solver_names = ['no_privacy',
                    'no_privacy_gw',
                    'no_privacy_ew',
                    'output_pert',
                    'output_pert_gw',
                    'output_pert_ew',
                    'obj_pert',
                    'obj_pert_gw',
                    'obj_pert_ew',
                    'dp_sgd',
                    'dp_sgd_gw']
    eps = 1.0
    lam = 10**(-1)
    gw = 0.1
    mb = 50
    k = 5

    from metric import auc_roc
    from metric import error
    from data import generate_synthetic_data
    import time
    X, Y, test_X, test_Y = generate_synthetic_data(1000, 10, 0.1)
    for loss in ['lr', 'huber']:
        for solver_name in solver_names:
            print(loss, solver_name)
            start = time.time()
            solver = Solver(loss, solver_name)
            if solver_name == 'dp_sgd_gw':
                w = solver.solve(X, Y, eps, lam, mb, k, gw)
            elif solver_name == 'dp_sgd':
                w = solver.solve(X, Y, eps, lam, mb, k)
            elif 'gw' in solver_name:
                w = solver.solve(X, Y, eps, lam, gw)
            else:
                w = solver.solve(X, Y, eps, lam)
            print(auc_roc(X @ w, Y), auc_roc(test_X @ w, test_Y))
            print(error(X @ w, Y), error(test_X @ w, test_Y))
            print("time:", time.time() - start)
