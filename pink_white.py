#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# [File Name]    : pink_white.py
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
 1.00 新規作成．
 2.00 ノイズの指定方法を変更．
      White NoiseのσとCorner周波数からPink Noiseを生成していたのを
      WhiteとFlickerのPSD[dB/Hz]から生成するようにした．
 3.00 2.00のインプリだとPSDwがFsサンプリングしたときのPSD高さなので
      Spice Simで求めたPSD高さになっていない．不適切．
      Spice SimのPSDwとノイズ帯域(f0)からσ(sigmaWhi)を算出するように変更．
 4.00 今までのやり方だとCorner周波数付近のノイズが過小になってしまうので，
      WhiteとPinkを別々に作って足し合わせるように変更．
'''


def main():
    Fspl     = 1          # [Hz] Sampling Freq
    Alpha    = .5         # Slope of Pink Noise(0.5でFlicker Noise)
    # L        = 2**14    # Down Sample後のデータ数(for fft)
    L        = 2**14 * 16 # Down Sample後のデータ数(for welch)
    t        = np.arange(0, 1024 + L/Fspl, 1 / Fspl)
    Pw_sim_dB = -6           # PSD by Spice Sim at white. (one-sided) [dB/Hz]
    F0      = 10          # Noise Band Width by Spice Sim
    Ffli    = 10**(-3)    # probing frequency of flicker noise
    Pf_sim_dB = 20          # PSD by Sim at Ffli. (one-sided) [dB/Hz]

    # beta = 10**(Pf_sim_dB/10) * Ffli # Flicker coefficiency

    # Spice Sim結果からWhite Noise成分のσを算出
    Pw_sim   = 10**(Pw_sim_dB / 10) # [V**2/Hz]
    sigmaWhi = np.sqrt(Pw_sim * F0 * np.pi / 2)  # White Noise σ．np.random.normalで使用

    # FsplでサンプリングしたときのWhite Noiseの高さ(Pw)
    Pw = np.pi * Pw_sim * F0 / Fspl

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
    # Spink = Swhi * pinkShapingTwoside(fre, Knee, Alpha)
    Spink = Swhi * pinkShaping(fre, Knee, Alpha)

    # %% Inverse-FFT
    Tpink_re_im = fftpack.ifft(Spink)
    Tpink  = np.real(Tpink_re_im) # 虚部(位相)の情報は現実世界のノイズには載らないので無視

    # %% fftでPSD (乱数Tpinkから2のべき乗個だけsliceして，FFT評価)
    # NFFT = 2**nextpow2minus1(len(Tpink))
    # Tfft = Tpink[-int(NFFT):]
    # Y_all = fftpack.fft(Tfft, NFFT)
    # Y_nyq = Y_all[0:int(NFFT/2)+1]
    # Pxx = np.abs(Y_nyq)**2 / (NFFT * Fspl)
    # Pxx[1:-1] = 2 * Pxx[1:-1]   # one-sided
    # freq = (1 / L) * np.arange(NFFT/2 + 1) * Fspl

    # %% Calculating PSD by welch
    NFFT = 2**nextpow2minus1(len(Tpink))
    Tfft = Tpink[-int(NFFT):]
    freq, Pxx = signal.welch(x=Tfft, fs=Fspl, window='boxcar',
                             nperseg=2**14, noverlap=None,  # 50%
                             nfft=None, detrend='constant',  # constant,linear
                             return_onesided=True)

    # %% plot
    plotFreq(Pxx, freq, Knee, Alpha, sigmaWhi, Fspl)


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


def pinkShaping(f, knee, alpha):
    """
    one-side, two-side表現のどちらにも適用可能．
    pinkShapingTwoside()だとf=knee付近の盛り上がりが再現できないためこちらを使う．
    FlickerとWhiteを別にする．Flickerはf=kneeを基準(x1倍)として計算．Whiteは1倍のまま．
    √(White**2 + Flicker**2)を返すことでコーナ周波数付近でも正しい倍率を返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, -Fs/2, -Fs/2+Δf, ..., -Δf]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :rtype: TYPE
    """
    desc_f   = np.ones_like(f)    # Flicker Noise Gain
    desc_all = np.ones_like(f)    # White + Flicker Noise Gain

    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    # f = knee [Hz]を基準(x1)にしてFlickerを生成．
    desc_f[1:] = (np.abs(f[1:]) / knee)**(-alpha)
    # WhiteとFlickerの√2乗和で全体のノイズシェイピング倍率を生成．
    desc_all[1:] = np.sqrt(1 + desc_f[1:]**2)
    return desc_all


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


def pinkShapingOneside2(f, knee, alpha):
    """
    pinkShapingOneside()だとf=knee付近の盛り上がりが再現できないためこちらを使う．
    FlickerとWhiteを別にする．Flickerはf=kneeを基準(x1倍)として計算．Whiteは1倍のまま．
    √(White**2 + Flicker**2)を返すことで適切な倍率を返す．
    :param TYPE f: FFT(fftfreq)の結果，返ってきた周波数．
                   ただし，f[-1] = -Fs/2 を Fs/2 へ置き換えたものを想定．
                   [0, Δf, 2Δf, ..., Fs/2-Δf, Fs/2]
    :param TYPE knee: Corner Freq [Hz]
    :param TYPE alpha: Slope of Pink Noise
    :rtype: TYPE
    """
    desc_f   = np.ones_like(f)
    desc_all = np.ones_like(f)
    # f[0] = 0なのでそのまま**(-alpha)すると0除算warningが出る．よって除外．
    # desc[1:][f[1:] < knee] = np.abs((f[1:][f[1:] < knee] / knee)**(-alpha))
    # f = knee [Hz]を基準(x1)にしてFlickerを生成．
    desc_f[1:] = (np.abs(f[1:]) / knee)**(-alpha)
    # WhiteとFlickerの√2乗和で全体のノイズシェイピング倍率を生成．
    desc_all[1:] = np.sqrt(1 + desc_f[1:]**2)
    return desc_all


def plotFreq(Pxx, freq, Knee, Alpha, sigmaWhi, Fspl):
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
    PSDtheo1 = 10 * np.log10(pinkShapingOneside(freq, Knee, Alpha)**2 * 2*sigmaWhi**2 / Fspl)
    ax1.semilogx(freq[1:], PSDtheo1[1:], 'r',label='noise model1')
    print('Theoretical PSDwhi noise floor = {0:.3f} [dB]'.format(PSDtheo1[-1]))

    PSDtheo2 = 10 * np.log10(pinkShaping(freq, Knee, Alpha)**2 * 2*sigmaWhi**2 / Fspl)
    ax1.semilogx(freq[1:], PSDtheo2[1:], '--', color='r', label='noise model2')
    print('Theoretical PSDwhi noise floor = {0:.3f} [dB]'.format(PSDtheo2[-1]))

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
    fig1.savefig('pink_white.png')
    plt.show(block=False)


if __name__ == '__main__':
    main()
