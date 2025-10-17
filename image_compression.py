import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr


plt.close('all')

def svd_auto(A):
    U, S, VT = np.linalg.svd(A)
    return U,S,VT

    

def image_rgb(file):
    image = Image.open(file)
    image_array = np.array(image).astype(np.float64)
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    return image, r,g,b

def channel_compression(c_mat, n):
    U,S,V=svd_auto(c_mat)
    compressed_channel = np.dot(U[:, :n], np.dot(np.diag(S[:n]), V[:n, :]))
    compressed_channel=np.clip(compressed_channel,0,255,).astype(np.uint8)

    return Image.fromarray(compressed_channel), S[:n]

def compression(r, g, b, n):
    r_comp, S_r = channel_compression(r, n)
    g_comp, S_g = channel_compression(g, n)
    b_comp, S_b = channel_compression(b, n)

    return r_comp, g_comp, b_comp

def get_new_image(r_channel, g_channel, b_channel, n):
    r_compressed, g_compressed, b_compressed = compression(r_channel, g_channel, b_channel, n)
    image_C = Image.merge("RGB", (r_compressed,g_compressed,b_compressed))
    return image_C

def display(original_image, compressed_images, k_list, ratios):
    plt.figure()
    plt.imshow(original_image)
    plt.title("Original image")
    plt.axis("off")
    
    for i, comp_im in enumerate(compressed_images):
        plt.figure()
        plt.imshow(comp_im)
        plt.axis("off")
        plt.title(f"Compressed image. k = {k_list[i]}")
    plt.show()

def show_singular_values(R, G, B):

    U_R, S_R, Vt_R = np.linalg.svd(R)
    U_G, S_G, Vt_G =np.linalg.svd(G)
    U_B, S_B, Vt_B =np.linalg.svd(B)
    
    
    plt.figure(figsize=(12, 6))
    plt.plot(S_R, label='Red channel', color='r')
    plt.plot(S_G, label='Green channel', color='g')
    plt.plot(S_B, label='Blue channel', color='b')
    plt.yscale('log')  
    plt.title("Three channels' singular values ")
    plt.ylabel('Singular values (log scale)')
    plt.legend()
    plt.grid()
    

def display_colors(r,g,b):
    fig, ax = plt.subplots(1,3, figsize=(16, 5))
    fig.suptitle("RGB Color Separation")
    
    ax[0].imshow(r,cmap="Reds_r")
    ax[1].imshow(g,cmap="Greens_r")
    ax[2].imshow(b,cmap="Blues_r")
    plt.tight_layout()

def get_size(A):
    m_A , n_A = A.size[0], A.size[1]
    size_A = m_A * n_A * 3
    return size_A,m_A,n_A

def psnr_val(original, compressed):
    original_array = np.array(original)
    compressed_array = np.array(compressed)
    return psnr(original_array, compressed_array, data_range=255)



def main(image_p,k_list):
    image, r_channel, g_channel, b_channel = image_rgb(image_p)
    display_colors(r_channel,g_channel,b_channel)
    show_singular_values(r_channel, g_channel, b_channel)

    size_A,m_A,n_A=get_size(image)

    compressed_images = []
    size_list = []
    
    ratios=[]
    psnr_vals=[]

    for n in k_list :
        print(n)
        
        size_C = 3 * (m_A * n + n + n * n_A)
        image_C=get_new_image(r_channel, g_channel, b_channel,n)
        compressed_images.append(image_C)

        size_list.append(size_C)
        ratio = round(100 - 100*size_C/size_A, 2)
        ratios.append(ratio)
        
        psnr_v=psnr_val(image,image_C)
        psnr_vals.append(psnr_v)
        

    display(image, compressed_images, k_list, ratios)
    
    return psnr_vals


image_p='./la_danse.jpg'
k_list=[5,50,500]
a=main(image_p,k_list)
