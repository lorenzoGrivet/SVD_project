from audio import *

audio_path="./10_secondi.wav"
ks=range(1,265)
l=[]

for n in ks:
    ll=[]
    print("Computing... n = ",n)
    S_magnitude,S_phase,y_original,sr =load_file(audio_path,16000)
    U,Sigma,VT=svd_auto(S_magnitude)
    S_magnitude_approx=np.dot(U[:, :n], np.dot(np.diag(Sigma[:n]), VT[:n, :]))
    S_approx = S_magnitude_approx * S_phase
    y_approx = librosa.istft(S_approx)
    
    pesq_v=calculate_pesq(sr,y_original,y_approx)
    ll.append(pesq_v)
    sdr=calculate_sdr(y_original,y_approx)
    ll.append(sdr)
    
    l.append(ll)

import csv

# Convertire i numeri decimali in stringhe con virgole come separatori
l_formatted = [
    [str(value).replace('.', ',') if isinstance(value, float) else value for value in row]
    for row in l
]

# Specifica il percorso del file CSV
csv_filename = 'output_audio_test.csv'

# Scrivi il file CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')

    # Scrivi le righe formattate nel file CSV
    writer.writerows(l_formatted)

    

