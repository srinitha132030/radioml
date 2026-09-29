"""
Generate sample signal files for testing the RadioML AMC system.
Run: python generate_samples.py
"""
import numpy as np
import pickle
import os

os.makedirs('sample_signals', exist_ok=True)

MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

def generate_iq_signal(mod, snr_db=0, n_samples=128):
    """Generate a synthetic I/Q signal for a given modulation type."""
    np.random.seed(42)
    t = np.linspace(0, 1, n_samples)
    noise_std = 10 ** (-snr_db / 20)
    noise = np.random.randn(2, n_samples) * noise_std

    if mod == 'BPSK':
        bits = np.random.choice([-1, 1], n_samples // 2)
        I = np.repeat(bits, 2)[:n_samples]
        Q = np.zeros(n_samples)
    elif mod == 'QPSK':
        bits_i = np.random.choice([-1, 1], n_samples // 2)
        bits_q = np.random.choice([-1, 1], n_samples // 2)
        I = np.repeat(bits_i, 2)[:n_samples]
        Q = np.repeat(bits_q, 2)[:n_samples]
    elif mod == '8PSK':
        phases = np.random.choice(np.linspace(0, 2*np.pi, 8, endpoint=False), n_samples)
        I = np.cos(phases)
        Q = np.sin(phases)
    elif mod == 'QAM16':
        levels = [-3, -1, 1, 3]
        I = np.random.choice(levels, n_samples) / 3.0
        Q = np.random.choice(levels, n_samples) / 3.0
    elif mod == 'QAM64':
        levels = np.arange(-7, 8, 2)
        I = np.random.choice(levels, n_samples) / 7.0
        Q = np.random.choice(levels, n_samples) / 7.0
    elif mod == 'PAM4':
        levels = [-3, -1, 1, 3]
        I = np.random.choice(levels, n_samples) / 3.0
        Q = np.zeros(n_samples)
    elif mod in ('CPFSK', 'GFSK'):
        freq = np.random.choice([-1, 1], n_samples)
        phase = np.cumsum(freq) * np.pi / 4
        I = np.cos(phase)
        Q = np.sin(phase)
    elif mod == 'AM-DSB':
        msg = np.sin(2 * np.pi * 2 * t)
        carrier_freq = 10
        I = (1 + 0.5 * msg) * np.cos(2 * np.pi * carrier_freq * t)
        Q = (1 + 0.5 * msg) * np.sin(2 * np.pi * carrier_freq * t)
    elif mod == 'AM-SSB':
        I = np.cos(2 * np.pi * 5 * t)
        Q = np.sin(2 * np.pi * 5 * t)
    elif mod == 'WBFM':
        msg = np.sin(2 * np.pi * 1 * t)
        phase = np.cumsum(msg) * 0.5
        I = np.cos(phase)
        Q = np.sin(phase)
    else:
        I = np.cos(2 * np.pi * 5 * t)
        Q = np.sin(2 * np.pi * 5 * t)

    return np.array([I[:n_samples] + noise[0], Q[:n_samples] + noise[1]])


print("Generating sample signal files...")

# 1. Generate .npy files (one per modulation)
for mod in MODULATIONS:
    sig = generate_iq_signal(mod, snr_db=10)  # shape: (2, 128)
    sig_batch = sig[np.newaxis, :, :]  # shape: (1, 2, 128)
    fname = f'sample_signals/{mod}_snr10.npy'
    np.save(fname, sig_batch)
    print(f'  Saved: {fname}')

# 2. Generate .csv files (I,Q in 256 columns)
for mod in ['BPSK', 'QPSK', 'QAM16']:
    sig = generate_iq_signal(mod, snr_db=5)  # (2, 128)
    combined = np.concatenate([sig[0], sig[1]])  # (256,)
    fname = f'sample_signals/{mod}_snr5.csv'
    np.savetxt(fname, combined.reshape(1, -1), delimiter=',', fmt='%.6f')
    print(f'  Saved: {fname}')

# 3. Generate a mini RadioML-format .pkl
mini_data = {}
for mod in MODULATIONS[:4]:  # Just first 4 for small size
    for snr in [0, 10]:
        samples = np.stack([generate_iq_signal(mod, snr) for _ in range(100)])  # (100, 2, 128)
        mini_data[(mod, snr)] = samples

fname = 'sample_signals/mini_radioml_sample.pkl'
with open(fname, 'wb') as f:
    pickle.dump(mini_data, f)
print(f'  Saved: {fname}')

# 4. Generate .txt files
for mod in ['BPSK', 'AM-DSB']:
    sig = generate_iq_signal(mod, snr_db=8)
    combined = np.concatenate([sig[0], sig[1]])
    fname = f'sample_signals/{mod}_snr8.txt'
    np.savetxt(fname, combined.reshape(1, -1), fmt='%.6f')
    print(f'  Saved: {fname}')

print(f'\n✅ Done! {len(os.listdir("sample_signals"))} sample files created in ./sample_signals/')
print('\nHow to use:')
print('  - Upload .npy files (e.g., BPSK_snr10.npy) in the User > Predict page')
print('  - Upload .csv files (e.g., QPSK_snr5.csv)')
print('  - Upload mini_radioml_sample.pkl as a mini test dataset')
