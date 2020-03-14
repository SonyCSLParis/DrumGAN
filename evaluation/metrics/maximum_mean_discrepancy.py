import torch

def compute_kernel(x, y, k):
    batch_size_x, dim_x = x.size()
    batch_size_y, dim_y = y.size()
    assert dim_x == dim_y

    xx = x.unsqueeze(1).expand(batch_size_x, batch_size_y, dim_x)
    yy = y.unsqueeze(0).expand(batch_size_x, batch_size_y, dim_y)
    distances = (xx - yy).pow(2).sum(2)
    return k(distances)

def mmd(z_tilde, z, kernel='imq'):
    # gaussian
    def gaussian(d, var=16.):
        return torch.exp(- d / var).sum(1).sum(0)

    # inverse multiquadratics
    def inverse_multiquadratics(d, var=16.):
        """
        :param d: (num_samples x, num_samples y)
        :param var:
        :return:
        """
        return (var / (var + d)).sum(1).sum(0)
    if kernel == 'imq':

        k = inverse_multiquadratics
    elif kernel == 'gaussian':
        k = gaussian
    else:
        raise AttributeError(f'Kernel type {kernel} not understood. Available: [gaussian, imq]')

    batch_size = z_tilde.size(0)
    zz_ker = compute_kernel(z, z, k)
    z_tilde_z_tilde_ker = compute_kernel(z_tilde, z_tilde, k)
    z_z_tilde_ker = compute_kernel(z, z_tilde, k)

    first_coefs = 1. / (batch_size * (batch_size - 1))
    second_coef = 2 / (batch_size * batch_size)
    mmd_distance = (first_coefs * zz_ker
           + first_coefs * z_tilde_z_tilde_ker
           - second_coef * z_z_tilde_ker)
    return mmd_distance