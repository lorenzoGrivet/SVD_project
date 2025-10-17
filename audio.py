import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pesq import pesq as pq
import os
import math
from mir_eval.separation import bss_eval_sources

def load_file(p,sr):
    y, sr = librosa.load(p, sr=sr)

    S = librosa.stft(y) #from time to frequency domain. rows are the frequencies, columns are the temporal segments
    S_mag,S_phase = librosa.magphase(S) 
    #mag is the module of the elemente(intensity of the signal, phase is the angle in every couple time-freq)
    
    # Converte il segnale dal dominio temporale a quello delle frequenze,
    # consentendo analisi dettagliate (ad esempio, identificare le frequenze dominanti in un dato momento).
    
    return S_mag,S_phase, y,sr

def plot_spectogram(S,sr,b,n):
    
    plt.figure(figsize=(10, 4))

    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    if b:
        plt.title(f'Compressed Spectrogram {n}')
    else:
        plt.title('Original Spectrogram')
        
    pass

def plot_singular(Sigma):
    plt.figure()
    plt.plot(Sigma)
    plt.axvline(x=264,color="red",linestyle="--")
    plt.title("Singular Values")
    plt.ylabel("Magnitude")
    plt.grid()



def svd_auto(S):
    U, Sigma, VT = np.linalg.svd(S)
    return U,Sigma,VT


def calculate_pesq(sr, original, compressed):
    score = pq(sr, original[:len(compressed)], compressed, "wb")
    return score

  #save the file cleaning the directory
def save_file(dir,y,sr,n):
    if n==-1:
        sf.write(os.path.join(dir, f"original.wav"), y, sr)
    else:
        sf.write(os.path.join(dir, f"compressed{n}.wav"), y, sr)
    pass

def clear_dir(dir):
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted file: {path}")
    pass


def calculate_sdr(original, compressed):
    original = original[:len(compressed)] 
    sdr, sir, sar, perm = bss_eval_sources(original[np.newaxis, :], compressed[np.newaxis, :])
    return sdr[0]


def main(audio_path,n_sv,dir):
    #deleet files from previous run
    clear_dir(dir)
      
    S_magnitude,S_phase,y_original,sr =load_file(audio_path,16000)
    
    plot_spectogram(S_magnitude,sr,False,n=0)
        
    U,Sigma,VT=svd_auto(S_magnitude)
    
    plot_singular(Sigma)
    
    m_A,n_A=S_magnitude.shape
    size_A=m_A*n_A  
    
    pesq_scores=[]
    sdr_scores=[]

    for nn in n_sv:
                    
        n_max=math.floor(m_A*n_A/(m_A+1+n_A))
                    
        if nn > n_max:
            #so the ratio is not negative
            n=n_max
            print("too less singular. n = ",n)
        else:
            n=nn
                    
        S_magnitude_approx=np.dot(U[:, :n], np.dot(np.diag(Sigma[:n]), VT[:n, :]))
        
        
        plot_spectogram(S_magnitude_approx,sr,True,n)
        
        #rebuilt compressed matrix multiplying by phase
        S_approx = S_magnitude_approx * S_phase

        size_approx = m_A * n + n + n * n_A
        print("compression ratio:", 1-size_approx/size_A )
        
        y_approx = librosa.istft(S_approx)
               
        #save file approx
        save_file(dir,y_approx,sr,n)
        
        # Calcola il PESQ
        pesq_score = calculate_pesq(sr, y_original, y_approx)
        pesq_scores.append(pesq_score)
               
        #sdr
        sdr_score=calculate_sdr(y_original,y_approx)
        sdr_scores.append(sdr_score)
        
    
    #save fil original
    save_file(dir,y_original,sr,n=-1)
    
    print("Pesq scores: ",pesq_scores)
        
    print("SDR scores: ",sdr_scores)
    



audio_path="./10_secondi.wav"
dir="./compressed"
n_sv=[1,100,300]

main(audio_path,n_sv,dir)
plt.show()
