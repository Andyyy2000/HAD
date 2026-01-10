import numpy as np
import torch
from numpy.linalg import norm, solve, inv

square_size = 100 / 1000
centers_norm = [[0, 0], [0, 0.05], [0, 0.1], [0.05, 0.05], [0.1, 0], [0.1, 0.1], [0, -0.05], [0, -0.1], [0.05, -0.05],
           [0.1, -0.1], [-0.05, 0.05], [-0.1, 0], [-0.1, 0.1], [-0.05, -0.05], [-0.1, -0.1]]
centers = np.array(centers_norm)
num_BS = 15
num_users = 8
agent_size = (1, 8)
num_classes = 3
tao_p = 20
tao_c = 140
tao_t = 200
tao_s = 40
sigma_sq = 3 ** 2
P_star = 10 * 1e-3
sigma_c_sq = 1
epsilon_reg = 1e-9
xi1 = 0.20
xi2 = 0.05
points = np.array([[-0.085, 0.056], [-0.012, 0.045], [0.095, 0.0077], [0.05, -0.086], [-0.046, 0], [0.0358, 0.0607], [-0.0238, -0.0868], [-0.0423, 0.0819]])
labels = np.array([1, 1, 1, 2, 1, 3, 1, 2])
N_r = 8
N_t = 8

def channel_model(points, centers):
    N_ap = len(centers)
    N_ue = len(points)
    distance_km = norm(centers[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    diff_vector = centers[:, np.newaxis, :] - points[np.newaxis, :, :]
    theta_rad = np.arctan2(diff_vector[:, :, 1], diff_vector[:, :, 0])
    distance_m = distance_km * 1000
    PL = 18.8 + 35 * np.log10(distance_km + 1e-6)
    beta_linear = 10 ** (-PL / 10)
    alpha = 9 * np.sqrt(beta_linear) * np.exp(-1j * 2 * np.pi * distance_m / 0.005)
    m_indices = np.arange(N_r)
    sin_theta = np.sin(theta_rad)
    a_bar = (1 / np.sqrt(N_r)) * np.exp(
        1j * np.pi * m_indices[np.newaxis, np.newaxis, :] * sin_theta[:, :, np.newaxis])
    channel_g = alpha[:, :, np.newaxis] * a_bar
    return channel_g, alpha, a_bar, theta_rad

def generate_pilot_sequences(N_ue, tau_p):
    if tau_p < N_ue:
        raise ValueError("导频长度 tau_p 必须大于或等于 UE 数量 N_ue 才能保证正交性。")
    W = np.fft.fft(np.eye(tau_p)) / np.sqrt(tau_p)
    pilot_sequences = W[:, :N_ue]  # 形状 (tau_p, N_ue)
    rho = [pilot_sequences[:, n].reshape(-1, 1) for n in range(N_ue)]
    return rho

def mmse_estimation_vectorized(channel_g, alpha, a_bar, rho_list, tau_p, P_star, sigma_sq, Nr):
    N_ap, N_ue, _ = channel_g.shape
    I_Nr = np.eye(Nr)
    channel_g_hat = np.zeros((N_ap, N_ue, Nr), dtype=complex)
    for k in range(N_ap):
        for n in range(N_ue):
            g_nk = channel_g[k, n, :].reshape(Nr, 1)
            alpha_nk = alpha[k, n]
            a_bar_nk = a_bar[k, n, :].reshape(Nr, 1)
            rho_nk = rho_list[n]  # (tau_p, 1)
            rho_nk_H = rho_nk.T.conjugate()
            N_nk_real = np.random.normal(0, np.sqrt(1 / 2), size=(Nr, tau_p))
            N_nk_imag = np.random.normal(0, np.sqrt(1 / 2), size=(Nr, tau_p))
            N_nk = N_nk_real + 1j * N_nk_imag  # 形状 (Nr, tau_p)
            Y_nk = np.sqrt(tau_p * P_star) * (g_nk @ rho_nk_H) + N_nk
            y_tilde_nk = Y_nk @ rho_nk  # 形状 (Nr, 1)
            a_bar_nk_H = a_bar_nk.T.conjugate()
            C_g_g = (np.abs(alpha_nk) ** 2) * (a_bar_nk @ a_bar_nk_H + sigma_sq * I_Nr)
            C_y_y_tilde = tau_p * P_star * C_g_g + I_Nr
            C_g_y_tilde = np.sqrt(tau_p * P_star) * C_g_g
            numerator = C_g_y_tilde @ y_tilde_nk
            g_hat = np.linalg.solve(C_y_y_tilde, numerator)
            channel_g_hat[k, n, :] = g_hat.flatten()
    return channel_g_hat

def calculate_sinr_matrix_for_ap(k, channel_g_hat, alpha_all, a_bar_all, P_c, sigma_sq, sigma_c_sq, Nr):
    _, N_ue, _ = channel_g_hat.shape
    I_Nr = np.eye(Nr)
    alpha_k = alpha_all[k, :]
    a_bar_k = a_bar_all[k, :, :]
    g_hat_k = channel_g_hat[k, :, :]
    P_k_vector = P_c[k, :]
    norm_g_hat_k = norm(g_hat_k, axis=1, keepdims=True)  # (N_ue, 1)
    W_k = np.divide(g_hat_k, norm_g_hat_k, out=np.zeros_like(g_hat_k, dtype=complex), where=norm_g_hat_k != 0)
    W_k_vector = W_k.reshape(N_ue, Nr, 1)  # (N_ue, Nr, 1)
    W_k_H = W_k_vector.conjugate().transpose(0, 2, 1)  # (N_ue, 1, Nr)
    A_bar_k_vector = a_bar_k.reshape(N_ue, Nr, 1)
    A_bar_k_H = A_bar_k_vector.conjugate().transpose(0, 2, 1)  # (N_ue, 1, Nr)
    A_bar_outer = A_bar_k_vector @ A_bar_k_H
    Alpha_sq = np.abs(alpha_k) ** 2  # (N_ue,)
    I_Nr_expanded = I_Nr[np.newaxis, :, :]  # (1, Nr, Nr)
    C_g_g_k = Alpha_sq[:, np.newaxis, np.newaxis] * (A_bar_outer + sigma_sq * I_Nr_expanded)
    Expected_Gain_sq = np.sum(A_bar_k_H * W_k_vector, axis=(1, 2))
    Desired_P = P_k_vector * Alpha_sq * np.abs(Expected_Gain_sq) ** 2
    Leaked_P = P_k_vector * Alpha_sq * sigma_sq
    Interf_Matrix_k = np.einsum('ci, bij, cj -> bc',
                                W_k.conj(),
                                C_g_g_k,
                                W_k,
                                optimize=True)
    Powered_Interf_Matrix_k = Interf_Matrix_k.real * P_k_vector[np.newaxis, :]
    mask = ~np.eye(N_ue, dtype=bool)
    Interference_P = np.sum(Powered_Interf_Matrix_k * mask, axis=1)
    Noise_P = sigma_c_sq
    Denominator = Leaked_P + Interference_P + Noise_P
    eta_k = Desired_P / Denominator
    return eta_k.real

def SINR(power, channel_g_hat, alpha, a_bar):
    sinr_matrix = np.zeros((1, num_users))
    for k in range(1):
        sinr_vector_k = calculate_sinr_matrix_for_ap(k, channel_g_hat, alpha, a_bar, power, sigma_sq, sigma_c_sq, N_r)
        sinr_matrix[k, :] = sinr_vector_k
    return sinr_matrix

def data_rate(sinr, a):
    a = (tao_c / tao_t) * a * 1e9
    rate = np.log2(1 + sinr)
    rate = a * rate
    return rate

def steering_vector_all(theta_matrix, N_ant):
    N_ap, N_ue = theta_matrix.shape
    m_indices = np.arange(N_ant).reshape(1, 1, N_ant, 1)
    sin_theta = np.sin(theta_matrix[:, :, np.newaxis, np.newaxis])
    phase_term = 1j * np.pi * m_indices * sin_theta
    a = (1 / np.sqrt(N_ant)) * np.exp(phase_term)
    return a

def kappa_factor_all(P_s_matrix, alpha_nk_all, G_jk):
    return G_jk * np.sqrt(P_s_matrix) * (alpha_nk_all**2)

def calculate_partial_u_all(theta_matrix, kappa_matrix, w_jk_t_matrix, Nr):
    Nt = Nr
    a_r = steering_vector_all(theta_matrix, Nr)
    a_t = steering_vector_all(theta_matrix, Nt)
    a_t_H = a_t.conjugate().transpose(0, 1, 3, 2)
    d = np.arange(Nt).reshape(1, 1, Nt, 1)
    kappa_exp = kappa_matrix[:, :, np.newaxis, np.newaxis]
    kappa_m_matrix = np.abs(kappa_matrix)
    kappa_m_exp = kappa_m_matrix[:, :, np.newaxis, np.newaxis]
    a_t_conj = a_t.conjugate()
    w_d_a_conj = w_jk_t_matrix * d * a_t_conj
    scalar_term1 = np.sum(w_d_a_conj, axis=2, keepdims=True)
    term1 = -a_r * scalar_term1
    scalar_term2 = a_t_H @ w_jk_t_matrix
    vector_factor2 = d * a_r
    term2 = scalar_term2 * vector_factor2  # (A, U, Nr, 1)
    d_u_d_theta = (1j * np.pi * kappa_exp / np.sqrt(Nr)) * (term1 + term2)
    common_factor = a_r * scalar_term2
    kappa_ratio = np.divide(kappa_exp, kappa_m_exp, out=np.zeros_like(kappa_exp, dtype=complex), where=kappa_m_exp != 0)
    d_u_d_kappa_m = kappa_ratio * common_factor * 1e6
    d_u_d_kappa_p = (1j * kappa_exp) * common_factor
    D_all = np.concatenate([d_u_d_theta, d_u_d_kappa_m, d_u_d_kappa_p], axis=3)
    return D_all

def calculate_fim_all(D_all, sigma_s_sq):
    D_H_all = D_all.conjugate().transpose(0, 1, 3, 2)
    fim_core_all = np.einsum('auik, aukl -> auil', D_H_all, D_all, optimize=True)
    FIM_all = (2 / sigma_s_sq) * np.real(fim_core_all)
    return FIM_all

def calculate_crlb_all(FIM_all, tau_s, tau_t, epsilon=1e-9):
    N_ap, N_ue, _, _ = FIM_all.shape
    FIM_safe = np.where(np.isnan(FIM_all) | np.isinf(FIM_all), 0.0, FIM_all)
    I_3x3 = np.eye(3)
    FIM_reg = FIM_safe + epsilon * I_3x3[np.newaxis, np.newaxis, :, :]
    try:
        FIM_inv_all = inv(FIM_reg)
    except np.linalg.LinAlgError:
        print("警告: 批量求逆失败。返回 NaN 矩阵。")
        return np.full((N_ap, N_ue), np.nan), np.full((N_ap, N_ue, 3, 3), np.nan)
    Tr_FIM_inv_all = np.trace(FIM_inv_all, axis1=2, axis2=3)
    omega_jk_t_matrix = (tau_s / tau_t) * Tr_FIM_inv_all
    return omega_jk_t_matrix, FIM_inv_all

def calculate_crlb_vectorized(theta_rad_matrix, alpha_nk_all, channel_g_hat_all, P_s_matrix, G_jk, Nr, Sigma_s_sq, Tau_s, Tau_t, EPSILON_REG):
    g_hat_all = channel_g_hat_all
    norm_g_hat_all = norm(g_hat_all, axis=2, keepdims=True)
    w_jk_t_matrix = np.divide(g_hat_all[:, :, :, np.newaxis],
                              norm_g_hat_all[:, :, np.newaxis] + 1e-9,
                              out=np.zeros_like(g_hat_all[:, :, :, np.newaxis], dtype=complex),
                              where=norm_g_hat_all[:, :, np.newaxis] != 0)
    kappa_matrix = kappa_factor_all(P_s_matrix, alpha_nk_all, G_jk)
    D_all = calculate_partial_u_all(theta_rad_matrix, kappa_matrix, w_jk_t_matrix, Nr)
    FIM_all = calculate_fim_all(D_all, Sigma_s_sq)
    crlb_matrix, fim_inv_all = calculate_crlb_all(FIM_all, Tau_s, Tau_t, epsilon=EPSILON_REG)
    return crlb_matrix

class CF_Mmimo(object):
    def __init__(self, num_agents, num_users):
        self.num_agents = num_agents
        self.state_matrix, self.sparse_matrix = self.sample_status()
        _, self.alpha_nk, self.a_bar_nk, self.theta_rad = channel_model(points, centers)

    def reset(self):
        self.state_matrix, self.sparse_matrix = self.sample_status()
        return self.state_matrix, self.sparse_matrix

    def step(self, high_command, low_actions):
        rewards = []
        for i in range(num_BS):
            low_level_state = self.state_matrix[i:i+1, :, :]
            low_level_action = low_actions[i:i+1, :]
            alpha = self.alpha_nk[i:i+1, :]
            a_bar = self.a_bar_nk[i:i+1, :, :]
            theta = self.theta_rad[i:i+1, :]
            agent_connection = high_command[i, :]
            reward = self._update_low_level_state(low_level_state, alpha, a_bar, theta, low_level_action, agent_connection)
            rewards.append([reward])
        low_level_rewards = rewards
        next_states, _ = self.sample_status()
        self.state_matrix = next_states

        return np.array(self.state_matrix), np.array(self.sparse_matrix), np.array(low_level_rewards)

    def _update_low_level_state(self, low_state, low_alpha, low_abar, low_theta, low_actions, high_command):
        action = low_actions * 1e-2
        sinr = SINR(action, low_state, low_alpha, low_abar)
        rate_raw = data_rate(sinr, action)
        rate = np.log2(rate_raw + 0.00001)
        rate_metric = rate * high_command
        crlb_raw = calculate_crlb_vectorized(low_theta, low_alpha, low_state, action, 0.06, N_r, sigma_sq, tao_s, tao_t, epsilon_reg)
        crlb = np.log2(crlb_raw + 0.00001)
        crlb_metric = crlb * high_command
        re = xi1*np.sum(rate_metric)-xi2*np.sum(crlb_metric)
        if np.sum(low_actions) > 50:
            re -= 0.2 * (np.sum(low_actions) - 50)
        return re

    @staticmethod
    def sample_status():
        channel_g, alpha_nk, a_bar_nk, theta_rad = channel_model(points, centers)
        rho_list = generate_pilot_sequences(num_users, tao_p)
        channel_g_hat = mmse_estimation_vectorized(channel_g, alpha_nk, a_bar_nk, rho_list, tao_p, P_star, sigma_sq, N_r)
        return channel_g_hat, labels


