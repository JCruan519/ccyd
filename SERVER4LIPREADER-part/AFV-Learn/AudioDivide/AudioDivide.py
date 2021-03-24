import numpy as np
import soundfile as sf
from scipy import fftpack
import scipy.signal
import scipy.interpolate
import peakutils.peak  # findpeaks
import matplotlib.pyplot as plt
import statsmodels.api as sm

def enframe(x , win , inc):
    nx = len(x)
    x = x.reshape(-1,1)
    # nwin = len(win)
    nwin = 1
    if nwin == 1:
        length = win
    else:
        length = nwin
    nf = np.floor((nx-length+inc)/inc)
    f = np.zeros((int(nf),length))
    indf = np.dot(inc ,(np.array(range(0,int(nf))).reshape(-1,1)))
    inds = np.array(range(1,length+1)).reshape(1,-1)
    indff , indss = indf[:,0] , inds[0,:]
    for i in range(length-1):
        indff = np.vstack((indff,indf[:,0]))
    for i in range(int(nf)-1):
        indss = np.vstack((indss,inds[0,:]))
    ind = indff.T + indss
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            f[i,j] = x[int(ind[i,j])-1,0]
    if (nwin > 1):
        w = win.reshape(1,-1)
        ww = w
        for i in range(int(nf)-1):
            ww = np.vstack((ww,w))
        f = f * ww
    return f

def vad_test(x):
    x = x.astype('float')
    x = x / np.max(np.abs(x))
    FrameLen = 240
    FrameInc = 80
    amp1 = 10
    amp2 = 0.1
    zcr1 = 15
    zcr2 = 1
    maxsilence = 8
    minlen = 15
    status = 0
    count = 0
    silence = 0

    tmp1 = enframe(x[0:-1,0],FrameLen,FrameInc)
    tmp2 = enframe(x[1:,0],FrameLen,FrameInc)
    signs = ((tmp1 * tmp2) < 0).astype('float')
    diffs = ((tmp1 - tmp2) > 0.02).astype('float')
    signsdiffs = signs * diffs
    zcr = sum(signsdiffs.T[:])
    aaa = scipy.signal.filtfilt([1,-0.9375],1,x.T, padtype='even', padlen=None, method='gust', irlen=None)
    amp = sum(np.abs(enframe(aaa.T,FrameLen,FrameInc)).T[:])

    amp1 = min(amp1,max(amp) / 4)
    amp2 = min(amp2,max(amp) / 8)

    x1 = 0
    x2 = 0

    for n in range(len(zcr)):
        goto = 0
        if status == 0 or status == 1:
            if amp[n] > amp1:
                x1 = max(n-count,1)
                status = 2
                silence = 0
                count = count + 1
            elif amp[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count = count + 1
            else:
                status = 0
                count = 0
        elif status == 2:
            if amp[n] > amp2 or zcr[n] > zcr2:
                count = count + 1
            else:
                silence = silence + 1
                if silence < maxsilence:
                    count += 1
                elif count < minlen:
                    status = 0
                    silence = 0
                    count = 0
                else:
                    status = 3
        else:
            break
    count = count - silence / 2
    x2 = x1 + count - 1
    return x1 , x2



def getQuantile(path):

    y, Fs = sf.read(path)

    y_raw = y
    y_raw_single = ((y_raw[:, 0] + y_raw[:, 1]) / 2).reshape(-1, 1)
    x1, x2 = vad_test(y_raw_single)
    x1_real = x1 * 80
    x2_real = x2 * 80
    q1 = x1_real
    q3 = x2_real

    single_left = y[:, 0]
    single_right = y[:, 1]

    mixed = single_left + single_right
    mixed = mixed / 2
    mixed_raw = mixed
    size_vector_raw = np.size(mixed)
    size_scaler_raw = size_vector_raw
    time_raw = np.linspace(0, size_scaler_raw, size_scaler_raw)

    t = 2
    ste = []
    zcc = []
    frameL = int(np.floor(Fs * t / 1000))

    for i in range(0, int(len(y) / frameL)):
        tmp = (y[i * frameL:(i + 1) * frameL, 0]).reshape(1, -1)
        ste.append(np.sum(tmp ** 2, axis=1).item())
        tmp2 = 0
        for j in range(0, len(tmp.T) - 1):
            if tmp[0, j] * tmp[0, j + 1] < 0:
                tmp2 += 1
        zcc.append(tmp2)
    ste = np.array(ste).reshape(1, -1)
    zcc = np.array(zcc).reshape(1, -1)

    t_e = 0.01
    t_z = 100

    vad = []
    for i in range(0, len(ste.T)):
        tmp = (ste[0, i] > t_e) * (zcc[0, i] < t_z)
        vad.append(int(tmp))
    vad = np.array(vad).reshape(1, -1)

    detect = np.ones((len(y), 1))

    for i in range(0, len(ste.T)):
        detect[i * frameL:(i + 1) * frameL, 0] = vad[0, i]

    mixed = []
    for i in range(0, len(y)):
        if detect[i, 0] == 1:
            mixed.append(y[i, 0])
    mixed = np.array(mixed).reshape(-1, 1)



    Amp_Factor = 10 / max(mixed)

    mixed = mixed * Amp_Factor

    Power = mixed * mixed

    size_vector = np.size(mixed)

    size_scaler = size_vector

    time = np.linspace(0, size_scaler, size_scaler)

    comp_env_mixed = scipy.signal.hilbert(np.squeeze(mixed)).reshape(-1, 1)

    env_mixed = abs(mixed + 1j * comp_env_mixed)

    downsmp_rate = 8

    graph_gain = 2
    graph_offset = 10

    smoothing_init_order = 300
    smoothing_step_order = 10
    smoothing_order = np.copy(smoothing_init_order)

    downsmp = np.linspace(1, size_scaler, int(size_scaler / downsmp_rate))
    time_dsmp = np.linspace(1, int(size_scaler / downsmp_rate), int(size_scaler / downsmp_rate))
    floor_downsmp = np.floor(downsmp).reshape(1, -1).astype(int)

    mixed_downsmp = []
    env_mixed_downsmp = []
    for i in np.squeeze(floor_downsmp):
        mixed_downsmp.append(mixed[i - 1, 0])
        env_mixed_downsmp.append(env_mixed[i - 1, 0])
    mixed_downsmp = np.array(mixed_downsmp).reshape(-1, 1)
    env_mixed_downsmp = np.array(env_mixed_downsmp).reshape(-1, 1)

    ###### 问题1：savgol_filter与lowess相差 0.0几
    env_mixed_dsmp_filtered = scipy.signal.savgol_filter(np.squeeze(env_mixed_downsmp), smoothing_init_order - 1,
                                                         1).reshape(-1, 1)

    threshold = 0.05

    for count in range(1, int(size_scaler / downsmp_rate)):
        if env_mixed_dsmp_filtered[count, 0] < threshold:
            env_mixed_dsmp_filtered[count, 0] = 0
    neg_env_mixed_dsmp_filtered = -env_mixed_dsmp_filtered

    ###### 问题2：找到的峰值个数相同，但是index不同
    minl = peakutils.peak.indexes(np.squeeze(neg_env_mixed_dsmp_filtered), min_dist=441)
    minv = []
    for i in range(len(minl)):
        minv.append(list(np.squeeze(neg_env_mixed_dsmp_filtered))[minl[i]])

    size_peak_vector = np.array(minv).shape

    peaks_num = 1
    emd_index = list(range(len(env_mixed_downsmp)))

    while (size_peak_vector[0] > peaks_num) and (smoothing_order <= 1800):

        smoothing_order = smoothing_order + smoothing_step_order

        env_mixed_dsmp_filtered = scipy.signal.savgol_filter(np.squeeze(env_mixed_downsmp), smoothing_order - 1,
                                                             1).reshape(-1, 1)

        threshold = 0.05

        for count in range(1, int(size_scaler / downsmp_rate)):
            if env_mixed_dsmp_filtered[count, 0] < threshold:
                env_mixed_dsmp_filtered[count, 0] = 0

        neg_env_mixed_dsmp_filtered = - env_mixed_dsmp_filtered

        minl = peakutils.peak.indexes(np.squeeze(neg_env_mixed_dsmp_filtered), min_dist=441)
        minv = []
        for i in range(len(minl)):
            minv.append(list(np.squeeze(neg_env_mixed_dsmp_filtered))[minl[i]])

        size_peak_vector = np.vstack((minv, minl)).T.shape

    minl_recover = minl * downsmp_rate

    q2 = q1 + minl_recover[int(len(minl_recover) / 2)]
    print(q1,q2,q3)

    return q1/size_scaler_raw , q2/size_scaler_raw , q3/size_scaler_raw

# path = 'C:/Users/11021\Desktop\AudioREC\matlabtopython/audioOutput/3d7aedea-19f9-4f5f-8d24-d7d7e2637130_qiye.wav'
# print(getQuantile(path))