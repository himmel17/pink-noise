#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# [File Name]    : pink_white_lpf.py
# ---------------------------------------------------------------------------
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
'''
[Purpose]
下記サイトのコードを参考にPink Noise(1/f)を生成．
https://gist.github.com/zonca/979729
データ数がLとFsplで決まるように変更を加えている．
[History]
 1.00 pink_white.pyのver3.00からfork.
      White Noiseの折り返しなし．高域(F0)でWhite Noise減衰．
'''


def main():
    Fspl      = 4          # [Hz] Sampling Freq
    Alpha     = .5         # Slope of Pink Noise(0.5でFlicker Noise)
    L         = 2**14 * 16 # データ数
    nperseg   = 2**14      # welchでスペクトル推定するときのセグメントデータ数(<=L)
    t         = np.arange(0, 1024 + L/Fspl, 1/Fspl)
    Pw_sim_dB = 20         # PSD by Spice Sim at white. (one-sided) [dB/Hz]
    F0        = 0.5        # Noise Band Width by Spice Sim
    Ffli      = 10**(-3)   # probing frequency of flicker noise
    Pf_sim_dB = 30         # PSD by Sim at Ffli. (one-sided) [dB/Hz]

    # Spice Sim結果からWhite Noise成分のσを算出
    Pw_sim   = 10**(Pw_sim_dB / 10) # [V**2/Hz]
    sigmaWhi = np.sqrt(Pw_sim * Fspl / 2)  # White Noise σ．np.random.normalで使用

    # FsplでサンプリングしたときのWhite Noiseの高さ(Pw)
    Pw = Pw_sim

    # コーナー周波数
    Pf_sim = 10**(Pf_sim_dB / 10) # [V**2/Hz]
    Knee = Pf_sim / Pw * Ffli     # Corner Freq

    print('sigmaWhi = {0:.3f}'.format(sigmaWhi))
    print('Knee = {0:e}'.format(Knee))

    # %% generate white noise in time domain
    wn = np.random.normal(0., sigmaWhi, len(t))

    # %% FFT
    Swhi = fftpack.fft(wn)
    fre = fftpack.fftfreq(len(t), d=1./Fspl)

    # %% shaping in freq domain
    Spink = Swhi * pinkLpfShaping(fre, Knee, Alpha, F0)

    # %% Inverse-FFT
    Tpink_re_im = fftpack.ifft(Spink)
    Tpink  = np.real(Tpink_re_im) # 虚部(位相)の情報は現実世界のノイズには載らないので無視

    # %% Calculating PSD by welch
    NFFT = 2**nextpow2minus1(len(Tpink))
    Tfft = Tpink[-int(NFFT):]
    freq, Pxx = signal.welch(x=Tfft, fs=Fspl, window='boxcar',
                             nperseg=nperseg, noverlap=None,  # 50% overlap
                             nfft=None, detrend='constant',  # constant,linear
                             return_onesided=True)

    # %% plot
    plotFreq(Pxx, freq, Knee, Alpha, sigmaWhi, Fspl, F0)


def nextpow2(L):
    """
    :param TYPE L: データ数
    :rtype: 指定値以上の最小の 2 のべき乗の指数(2**m_iはL以上になる)
    """
    m_f = np.log2(L)
    m_i = np.ceil(m_f)
    return int(m_i)


def nextpow2minus1(L):
    """
    :param TYPE L: データ数
    :rtype: 指定値以下の最大の 2 のべき乗の指数(2**m_iはL以下になる)
    """
    m_f = np.log2(L)
    m_i = np.floor(m_f)
    return int(m_i)


def pinkShapingTwoside(f, knee, alpha):
    """
    kneeよりも低周波は1/f**(-alpha)，kneeよりも高周波は1となる1d-arrayを返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, -Fs/2, -Fs/2+Δf, ..., -Δf]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :rtype: TYPE
    """
    desc = np.ones_like(f)
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    # for positive frequency
    desc[1:][(f[1:] < knee) & (f[1:] > 0)] = (f[1:][(f[1:] < knee) & (f[1:] > 0)] / knee)**(-alpha)
    # for negative frequency
    desc[1:][(f[1:] > -knee) & (f[1:] < 0)] = np.abs((f[1:][(f[1:] > -knee) & (f[1:] < 0)]) / knee)**(-alpha)
    return desc


def pinkShapingOneside(f, knee, alpha):
    """
    Plot時のNoise Model線を作るための関数．
    kneeよりも低周波は1/f**(-alpha)，kneeよりも高周波は1となる1d-arrayを返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   ただし，f[-1] = -Fs/2 を Fs/2 へ置き換えたものを想定．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, Fs/2]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :rtype: TYPE
    """
    desc = np.ones_like(f)
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    desc[1:][f[1:] < knee] = np.abs((f[1:][f[1:] < knee] / knee)**(-alpha))
    return desc


def pinkLpfShaping(f, knee, alpha, F0):
    """
    one-side, two-side表現のどちらにも適用可能．
    pinkShapingTwoside()だとf=knee付近の盛り上がりが再現できないためこちらを使う．
    kneeよりも低周波は1/f**(-alpha)，kneeよりも高周波は1となる1d-arrayを返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, -Fs/2, -Fs/2+Δf, ..., -Δf]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :param TYPE F0: Cut-off freq of LPF [Hz]
    :rtype: TYPE
    """
    # LPF shaping
    desc_lpf = F0 / np.sqrt(F0**2 + f**2)

    desc_f   = np.ones_like(f)
    desc_all = np.ones_like(f)

    # Pink shaping
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    # f = knee [Hz]を基準(x1)にしてFlickerを生成．
    desc_f[1:] = (np.abs(f[1:]) / knee)**(-alpha)
    # White w/ LPFとFlickerの√2乗和で全体のノイズシェイピング倍率を生成．
    desc_all[1:] = np.sqrt(desc_lpf[1:]**2 + desc_f[1:]**2)
    return desc_all


def pinkLpfShapingTwoside(f, knee, alpha, F0):
    """
    kneeよりも低周波は1/f**(-alpha)，kneeよりも高周波は1となる1d-arrayを返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, -Fs/2, -Fs/2+Δf, ..., -Δf]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :param TYPE F0: Cut-off freq of LPF [Hz]
    :rtype: TYPE
    """
    # LPF shaping
    desc = np.ones_like(f)
    desc = F0 / np.sqrt(F0**2 + f**2)

    # Pink shaping
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    # for positive frequency
    desc[1:][(f[1:] < knee) & (f[1:] > 0)] = (f[1:][(f[1:] < knee) & (f[1:] > 0)] / knee)**(-alpha)
    # for negative frequency
    desc[1:][(f[1:] > -knee) & (f[1:] < 0)] = np.abs((f[1:][(f[1:] > -knee) & (f[1:] < 0)]) / knee)**(-alpha)

    return desc


def pinkLpfShapingOneside(f, knee, alpha, F0):
    """
    Plot時のNoise Model線を作るための関数．
    kneeよりも低周波は1/f**(-alpha)，kneeよりも高周波は1となる1d-arrayを返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, -Fs/2, -Fs/2+Δf, ..., -Δf]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :param TYPE F0: Cut-off freq of LPF [Hz]
    :rtype: TYPE
    """
    desc = np.ones_like(f)
    # LPF shaping
    desc = F0 / np.sqrt(F0**2 + f**2)
    # Pink shaping
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    desc[1:][f[1:] < knee] = np.abs((f[1:][f[1:] < knee] / knee)**(-alpha))
    return desc


def plotFreq(Pxx, freq, Knee, Alpha, sigmaWhi, Fspl, F0):
    """
    DC(freq[0], Pxx[0])は除外してプロットする．
    """
    # plt.close('all')
    fig1 = plt.figure(num=1)
    plt.clf()
    ax1 = fig1.add_subplot(111)

    # %% measured
    PSDpink = 10 * np.log10(Pxx)
    ax1.semilogx(freq[1:], PSDpink[1:], label='sim')
    Pxx_whi = Pxx[freq > Knee]
    PSDpink_whi_ave = 10 * np.log10(np.sum(Pxx_whi) / len(Pxx_whi))
    print('Measured PSD in white noise = {0:.3f} [dB]'.format(PSDpink_whi_ave))

    # %% theory
    PSDtheo1 = 10 * np.log10(pinkLpfShapingOneside(freq, Knee, Alpha, F0)**2 * 2*sigmaWhi**2 / Fspl)
    ax1.semilogx(freq[1:], PSDtheo1[1:], 'r',label='noise model1')
    # print('Theoretical PSDwhi noise floor = {0:.3f} [dB]'.format(PSDtheo1[-1]))

    PSDtheo2 = 10 * np.log10(pinkLpfShaping(freq, Knee, Alpha, F0)**2 * 2*sigmaWhi**2 / Fspl)
    ax1.semilogx(freq[1:], PSDtheo2[1:], '--', color='r',label='noise model2')

    # %% decoration
    ax1.vlines(Knee, *plt.ylim(), linestyles='dashed', lw=1)
    ax1.set_xlabel('Freq [Hz]', fontsize=14)
    ax1.set_ylabel('PSD (one-sided) [dB/Hz]', fontsize=14)
    ax1.set_ylim([0, 40])
    # ax1.set_title('Amplitude spectrum')
    ax1.legend(loc='best')
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle=':')
    ax1.grid(b=True, which='minor', color='k', linestyle=':', alpha=0.3)

    fig1.tight_layout()
    fig1.savefig('pink_white_lpf.png')
    plt.show(block=False)


if __name__ == '__main__':
    main()
