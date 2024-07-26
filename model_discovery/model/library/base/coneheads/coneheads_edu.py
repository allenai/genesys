import torch
import torch.nn.functional as F


def map_psi(x, r):
    x_x = x[..., :-1]
    x_y = F.sigmoid(x[..., -1])
    return x_x * x_y.unsqueeze(-1) * r, x_y * r


def penumbral(q, k, r=1, gamma=1, eps=1e-6):
    q_x, q_y = map_psi(q, r)
    k_x, k_y = map_psi(k, r)
    q_y = q_y.unsqueeze(2)
    k_y = k_y.unsqueeze(1)

    x_q_y = torch.sqrt(r**2 - q_y**2 + eps)
    x_k_y = torch.sqrt(r**2 - k_y**2 + eps)

    pairwise_dist = torch.cdist(q_x, k_x)

    lca_height = torch.maximum(torch.maximum(q_y**2, k_y**2),
                               r**2 - ((x_q_y + x_k_y - pairwise_dist) / 2)**2)

    lca_height_outcone = ((pairwise_dist**2 + k_y**2 - q_y**2) /
                          (2 * pairwise_dist + eps))**2 + q_y**2

    exists_cone = torch.logical_or(pairwise_dist <= x_q_y,
                                   (pairwise_dist - x_q_y)**2 + k_y**2 <= r**2)

    return -gamma * torch.where(exists_cone, lca_height, lca_height_outcone)


def map_xi(x):
    x_x = x[..., :-1]
    x_y = torch.exp(x[..., -1] / x.shape[-1])
    return x_x * x_y.unsqueeze(-1), x_y


def umbral(q, k, r=1, gamma=1):
    q_x, q_y = map_xi(q)
    k_x, k_y = map_xi(k)
    q_y = q_y.unsqueeze(2)
    k_y = k_y.unsqueeze(1)
    out = torch.maximum(torch.maximum(q_y, k_y),
                        (torch.cdist(q_x, k_x) / torch.sinh(torch.tensor(r)) +
                         torch.add(q_y, k_y)) / 2)
    return -gamma * out


if __name__=='__main__':
    # [b, n, d]
    q = torch.randn(100, 10, 2)
    k = torch.randn(100, 10, 2)

    # [100, 10, 10]
    print(umbral(q, k).shape)
    # [100, 10, 10]
    print(penumbral(q, k).shape)
