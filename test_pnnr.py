from image_compression import *

def svd_short(A_color):
    U,S,V = svd_auto(A_color)
    return U,S,V

def compression_short(U,S,V,n):
    compressed_channel = np.dot(U[:, :n], np.dot(np.diag(S[:n]), V[:n, :]))
    compressed_channel=np.clip(compressed_channel,0,255,).astype(np.uint8)

    return Image.fromarray(compressed_channel)

def channels_n(U,S,V,n):
    c_comp = compression_short(U,S,V, n)
    return c_comp



image='./la_danse_enorme.jpg'
ks=range(1,801)
psnr_vals=[]

im,r,g,b=image_rgb(image)
U_r,S_r,V_r = svd_short(r)
U_g,S_g,V_g = svd_short(g)
U_b,S_b,V_b = svd_short(b)


for n in ks:    
    r_c = channels_n(U_r,S_r,V_r,n)
    g_c=channels_n(U_g,S_g,V_g,n)
    b_c=channels_n(U_b,S_b,V_b,n)
    image_C = Image.merge("RGB", (r_c,g_c,b_c))
    
    print("Computing... n = ",n)
    p_val=psnr_val(im,image_C)
    print(p_val)
    psnr_vals.append(p_val)


import csv

# Convertire i numeri decimali in stringhe con virgole come separatori
l_formatted = [[str(value).replace('.', ',')] for value in psnr_vals]

# Specifica il percorso del file CSV
csv_filename = 'output_imm_test2.csv'

# Scrivi il file CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')

    # Scrivi ogni numero su una riga separata
    writer.writerows(l_formatted)
