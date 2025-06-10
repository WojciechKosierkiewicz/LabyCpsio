import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


pictures_folder = r'C:\Users\wojte\Documents\LabyCpsio\lab1\obrazy'
images = [f for f in os.listdir(pictures_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
selected_image = None

def update_image(*args):
    global selected_image
    selected_file = image_selector.get()
    if selected_file:
        image_path = os.path.join(pictures_folder, selected_file)
        image = Image.open(image_path).convert('L')  
        
        width, height = image.size
        if width > 500 or height > 500:
            ratio = min(500/width, 500/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height))
            
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo 
        selected_image = np.array(image)

def show_profile_lines():
    if selected_image is None:
        return
        
    row = int(row_scale.get())
    col = int(col_scale.get())
    
    display_image = Image.fromarray(selected_image)
    display_image = display_image.convert('RGB')  
    
    for x in range(display_image.width):
        display_image.putpixel((x, row), (255, 0, 0))  
        
    for y in range(display_image.height):
        display_image.putpixel((col, y), (0, 255, 0))  
    
    photo = ImageTk.PhotoImage(display_image)
    image_label.configure(image=photo)
    image_label.image = photo

def update_profiles(*args):
    if selected_image is None:
        return
        
    height, width = selected_image.shape
    row_scale.configure(to=height-1)
    col_scale.configure(to=width-1)
    
    row = int(row_scale.get())
    if row < height:
        plt.figure("Szarosc pozioma")
        plt.clf()
        plt.plot(selected_image[row,:])
        plt.title(f'Szarosc pozioma(wiersz {row})')
        plt.xlabel("Pozycja x")
        plt.ylabel("Poziom szarości")
        plt.grid(True)
        plt.pause(0.01)
    
    col = int(col_scale.get())
    if col < width:
        plt.figure("Szarosc pionowa")
        plt.clf()
        plt.plot(selected_image[:,col])
        plt.title(f'Szarosc pionowa(kolumna {col})')
        plt.xlabel("Pozycja y")
        plt.ylabel("Poziom szarości")
        plt.grid(True)
        plt.pause(0.01)

def multiply_by_constant():
    global selected_image
    if selected_image is None:
        return
        
    try:
        c = float(multiply_c_entry.get())
        img_float = selected_image.astype(np.float32)
        transformed = c * img_float
        
        if transformed.max() > 0:  
            transformed = (transformed * 255 / transformed.max()).clip(0, 255)
        
        selected_image = transformed.astype(np.uint8)
        
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        update_profiles()
        show_profile_lines()
    except ValueError:
        print("Please enter a valid number for constant c")

def logarithmic_transform():
    global selected_image
    if selected_image is None:
        return
        
    try:
        c = float(log_c_entry.get())
        img_float = selected_image.astype(np.float32)
        
        transformed = c * np.log1p(img_float)
        
        if transformed.max() > 0:  
            transformed = (transformed * 255 / transformed.max()).clip(0, 255)
            
        selected_image = transformed.astype(np.uint8)
        
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        update_profiles()
        show_profile_lines()
    except ValueError:
        print("Please enter a valid number for constant c")

def contrast_transform():
    global selected_image
    if selected_image is None:
        return
        
    try:
        m = float(contrast_m_entry.get())  
        e = float(contrast_e_entry.get())
        
        img_float = selected_image.astype(np.float32) / 255.0
        
        epsilon = 1e-10
        
        ratio = m / (img_float + epsilon)
        ratio = np.clip(ratio, 0, 100)
        
        transformed = 1.0 / (1.0 + ratio ** e)
        transformed = (transformed * 255.0).clip(0, 255).astype(np.uint8)
        selected_image = transformed
        
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        
        update_profiles()
        show_profile_lines()
        
    except ValueError:
        print("Please enter valid numbers for parameters m and e")

def gamma_correction():
    global selected_image
    if selected_image is None:
        return
        
    try:
        c = float(gamma_c_entry.get())
        gamma = float(gamma_gamma_entry.get())
        
        img_float = selected_image.astype(np.float32) / 255.0
        
        transformed = c * (img_float ** gamma)
        
        transformed = (transformed * 255.0).clip(0, 255).astype(np.uint8)
        selected_image = transformed
        
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        update_profiles()
        show_profile_lines()
        
    except ValueError:
        print("Please enter valid numbers for parameters c and gamma")

def histogram_equalization():
    global selected_image
    if selected_image is None:
        return
        
    hist, bins = np.histogram(selected_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    
    lookup_table = cdf_normalized.astype('uint8')
    
    original_img = selected_image.copy()
    
    selected_image = lookup_table[selected_image]
    
    display_image = Image.fromarray(selected_image)
    photo = ImageTk.PhotoImage(display_image)
    image_label.configure(image=photo)
    image_label.image = photo
    
    update_profiles()
    show_profile_lines()
    
    plt.figure("Porównanie histogramów")
    plt.clf()
    
    plt.subplot(2, 1, 1)
    plt.hist(original_img.flatten(), bins=256, range=[0, 256], color='b', alpha=0.7)
    plt.title("Orginalny histogram")
    plt.xlabel("Poziom szarości")
    plt.ylabel("częstotliwość")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.hist(selected_image.flatten(), bins=256, range=[0, 256], color='g', alpha=0.7)
    plt.title("Wyrównany histogram")
    plt.xlabel("Poziom szarości")
    plt.ylabel("częstotliwość")
    plt.grid(True)
    
    plt.tight_layout()
    plt.pause(0.01)

def apply_mean_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.mean(window)
            
    selected_image = result.astype(np.uint8)
    update_display()

def apply_median_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.median(window)
            
    selected_image = result.astype(np.uint8)
    update_display()

def apply_min_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.min(window)
            
    selected_image = result.astype(np.uint8)
    update_display()

def apply_max_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.max(window)
            
    selected_image = result.astype(np.uint8)
    update_display()

def update_display():
    display_image = Image.fromarray(selected_image)
    photo = ImageTk.PhotoImage(display_image)
    image_label.configure(image=photo)
    image_label.image = photo
    update_profiles()
    show_profile_lines()

def apply_sobel_horizontal():
    global selected_image
    if selected_image is None:
        return
        
    mask = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
                    
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_sobel_vertical():
    global selected_image
    if selected_image is None:
        return
        
    mask = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
                    
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_sobel_diagonal():
    global selected_image
    if selected_image is None:
        return
        
    mask = np.array([[0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]])
                    
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_laplacian():
    global selected_image
    if selected_image is None:
        return
        
    mask = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
                    
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
    
    result = selected_image.astype(float) - result
    result = result.clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_unsharp_masking():
    global selected_image
    if selected_image is None:
        return
        
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    blurred = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            blurred[i, j] = np.mean(window)
    
    mask = selected_image.astype(float) - blurred
    k = 2.0  
    result = selected_image.astype(float) + k * mask
    
    selected_image = result.clip(0, 255).astype(np.uint8)
    update_display()

def apply_high_boost():
    global selected_image
    if selected_image is None:
        return
        
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    blurred = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            blurred[i, j] = np.mean(window)
    
    A = 1.5  
    mask = selected_image.astype(float) - blurred
    result = A * selected_image.astype(float) + mask
    
    selected_image = result.clip(0, 255).astype(np.uint8)
    update_display()

def create_gaussian_kernel(size, sigma=1.0):
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    return kernel / np.sum(kernel)

def apply_gaussian_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    sigma = size / 6.0  
    kernel = create_gaussian_kernel(size, sigma)
    
    pad = size // 2
    padded = np.pad(selected_image.astype(float), pad, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+size, j:j+size]
            result[i, j] = np.sum(window * kernel)
            
    selected_image = result.clip(0, 255).astype(np.uint8)
    update_display()

def apply_multi_step_enhancement():
    global selected_image
    if selected_image is None:
        return
        
    original = selected_image.copy()
    
    steps_window = tk.Toplevel()
    steps_window.title("Multi-step Enhancement Process")
    steps_window.minsize(400, 300)
    
    canvas = tk.Canvas(steps_window)
    v_scrollbar = ttk.Scrollbar(steps_window, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(steps_window, orient="horizontal", command=canvas.xview)
    
    content_frame = ttk.Frame(canvas)
    
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    content_frame.bind("<Configure>", on_configure)
    
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    
    def show_step(img, title, row, col):
        display_img = Image.fromarray(img)
        if display_img.size[0] > 300:
            ratio = 300 / display_img.size[0]
            new_size = (300, int(display_img.size[1] * ratio))
            display_img = display_img.resize(new_size)
        photo = ImageTk.PhotoImage(display_img)
        label = ttk.Label(content_frame, image=photo)
        label.image = photo  
        label.grid(row=row, column=col, padx=5, pady=5)
        ttk.Label(content_frame, text=title).grid(row=row+1, column=col, padx=5, pady=(0,10))
    
    show_step(original, "a) Obraz wejściowy", 0, 0)
    
    mask = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
    padded = np.pad(original.astype(float), 1, mode='reflect')
    laplacian = np.zeros_like(original, dtype=float)
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            window = padded[i:i+3, j:j+3]
            laplacian[i, j] = np.sum(window * mask)
    laplacian_result = np.abs(laplacian).clip(0, 255).astype(np.uint8)
    show_step(laplacian_result, "b) Laplasjan", 0, 1)
    
    sum_result = (original.astype(float) + laplacian_result.astype(float)).clip(0, 255).astype(np.uint8)
    show_step(sum_result, "c) Suma a + b", 0, 2)
    
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    gradient_x = np.zeros_like(original, dtype=float)
    gradient_y = np.zeros_like(original, dtype=float)
    
    padded = np.pad(sum_result.astype(float), 1, mode='reflect')
    
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            window = padded[i:i+3, j:j+3]
            gradient_x[i, j] = np.abs(np.sum(window * sobel_x))
            gradient_y[i, j] = np.abs(np.sum(window * sobel_y))
    
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = gradient * 4.0  
    gradient = (gradient * 255 / gradient.max()).clip(0, 255).astype(np.uint8)
    gradient[gradient < 15] = 0  
    show_step(gradient, "d) Sobel Gradient", 2, 0)
    
    mean_result = np.zeros_like(gradient, dtype=float)
    padded_grad = np.pad(gradient.astype(float), 2, mode='reflect')
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            window = padded_grad[i:i+5, j:j+5]
            mean_result[i, j] = np.mean(window)
    mean_result = mean_result.clip(0, 255).astype(np.uint8)
    show_step(mean_result, "e) Mean Filter 5x5", 2, 1)
    
    combined = (mean_result.astype(float) * laplacian_result.astype(float)) / 255.0
    final_result = original.astype(float) + combined
    
    gamma = 0.6  
    c = 1.2     
    final_result = final_result / final_result.max()  
    final_result = c * (final_result ** gamma)
    final_result = (final_result * 255).clip(0, 255).astype(np.uint8)
    show_step(final_result, "f) Final Result", 2, 2)
    
    selected_image = final_result
    update_display()
    
    h_scrollbar.pack(side="bottom", fill="x")
    v_scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    close_btn = ttk.Button(steps_window, text="Close", command=steps_window.destroy)
    close_btn.pack(side="bottom", pady=5)


root = tk.Tk()
root.title("Przetwarzanie obrazów")

def on_closing():
    plt.close('all')  
    image_window.destroy()  
    root.quit()  
    root.destroy()  

root.protocol("WM_DELETE_WINDOW", on_closing)


canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)


scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)


def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


root.minsize(600, 400)
canvas.configure(width=580)  


canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

main_frame = ttk.Frame(scrollable_frame, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)


image_window = tk.Toplevel()
image_window.title("Podgląd obrazu")


ttk.Label(main_frame, text="Wybierz wiersz:").pack(pady=5)
row_scale = tk.Scale(main_frame, from_=0, to=500, orient='horizontal')
row_scale.pack(fill='x', padx=10)

ttk.Label(main_frame, text="Wybierz kolumnę:").pack(pady=5) 
col_scale = tk.Scale(main_frame, from_=0, to=500, orient='horizontal')
col_scale.pack(fill='x', padx=10)

transform_frame = ttk.LabelFrame(main_frame, text="Transformacje punktowe", padding="5")
transform_frame.pack(pady=10, fill='x')

multiply_frame = ttk.LabelFrame(transform_frame, text="a) Mnożenie przez stałą", padding="5")
multiply_frame.pack(pady=5, fill='x', padx=5)

ttk.Label(multiply_frame, text="c:").pack(side='left', padx=5)
multiply_c_entry = ttk.Entry(multiply_frame, width=8)
multiply_c_entry.insert(0, "1.0")
multiply_c_entry.pack(side='left', padx=5)

multiply_btn = ttk.Button(multiply_frame, text="Zastosuj T(r) = c·r", command=multiply_by_constant)
multiply_btn.pack(side='left', padx=5)

log_frame = ttk.LabelFrame(transform_frame, text="b) Transformacja logarytmiczna", padding="5")
log_frame.pack(pady=5, fill='x', padx=5)

ttk.Label(log_frame, text="c:").pack(side='left', padx=5)
log_c_entry = ttk.Entry(log_frame, width=8)
log_c_entry.insert(0, "1.0")
log_c_entry.pack(side='left', padx=5)

log_btn = ttk.Button(log_frame, text="Zastosuj T(r) = c·log(1+r)", command=logarithmic_transform)
log_btn.pack(side='left', padx=5)

contrast_frame = ttk.LabelFrame(transform_frame, text="c) Transformacja kontrastu", padding="5")
contrast_frame.pack(pady=5, fill='x', padx=5)

param_frame = ttk.Frame(contrast_frame)
param_frame.pack(side='left', padx=5)

ttk.Label(param_frame, text="m:").grid(row=0, column=0, padx=5)
contrast_m_entry = ttk.Entry(param_frame, width=8)
contrast_m_entry.insert(0, "0.5")  
contrast_m_entry.grid(row=0, column=1, padx=5)

ttk.Label(param_frame, text="e:").grid(row=0, column=2, padx=5)
contrast_e_entry = ttk.Entry(param_frame, width=8)
contrast_e_entry.insert(0, "4")
contrast_e_entry.grid(row=0, column=3, padx=5)

contrast_btn = ttk.Button(contrast_frame, text="Zastosuj T(r) = 255/(1+(m/r)^e)", command=contrast_transform)
contrast_btn.pack(side='left', padx=5)

gamma_frame = ttk.LabelFrame(transform_frame, text="d) Korekcja gamma", padding="5")
gamma_frame.pack(pady=5, fill='x', padx=5)

param_frame_gamma = ttk.Frame(gamma_frame)
param_frame_gamma.pack(side='left', padx=5)

ttk.Label(param_frame_gamma, text="c:").grid(row=0, column=0, padx=5)
gamma_c_entry = ttk.Entry(param_frame_gamma, width=8)
gamma_c_entry.insert(0, "1.0")
gamma_c_entry.grid(row=0, column=1, padx=5)

ttk.Label(param_frame_gamma, text="γ:").grid(row=0, column=2, padx=5)
gamma_gamma_entry = ttk.Entry(param_frame_gamma, width=8)
gamma_gamma_entry.insert(0, "2.2")
gamma_gamma_entry.grid(row=0, column=3, padx=5)

gamma_btn = ttk.Button(gamma_frame, text="Zastosuj s = c·r^γ", command=gamma_correction)
gamma_btn.pack(side='left', padx=5)

hist_eq_frame = ttk.LabelFrame(transform_frame, text="e) Wyrównanie histogramu", padding="5")
hist_eq_frame.pack(pady=5, fill='x', padx=5)

hist_eq_btn = ttk.Button(hist_eq_frame, text="Zastosuj wyrównanie histogramu", command=histogram_equalization)
hist_eq_btn.pack(padx=5, pady=5)

filters_frame = ttk.LabelFrame(main_frame, text="Filtry", padding="5")
filters_frame.pack(pady=10, fill='x')

lowpass_frame = ttk.LabelFrame(filters_frame, text="Filtry dolnoprzepustowe", padding="5")
lowpass_frame.pack(pady=5, fill='x', padx=5)

filter_sizes = ['3x3', '5x5', '7x7', '9x9', '11x11']
size_var = tk.StringVar(value='3x3')
ttk.Label(lowpass_frame, text="Rozmiar maski:").pack(side='left', padx=5)
size_combo = ttk.Combobox(lowpass_frame, values=filter_sizes, textvariable=size_var, state='readonly', width=6)
size_combo.pack(side='left', padx=5)

mean_btn = ttk.Button(lowpass_frame, text="Filtr średni",
                     command=lambda: apply_mean_filter(int(size_var.get()[0])))
mean_btn.pack(side='left', padx=5)

gaussian_btn = ttk.Button(lowpass_frame, text="Filtr Gaussa",
                         command=lambda: apply_gaussian_filter(int(size_var.get()[0])))
gaussian_btn.pack(side='left', padx=5)

other_filters_frame = ttk.LabelFrame(filters_frame, text="Inne filtry", padding="5")
other_filters_frame.pack(pady=5, fill='x', padx=5)

median_btn = ttk.Button(other_filters_frame, text="Filtr medianowy",
                       command=lambda: apply_median_filter(int(size_var.get()[0])))
median_btn.pack(side='left', padx=5)

min_btn = ttk.Button(other_filters_frame, text="Filtr minimalny",
                    command=lambda: apply_min_filter(int(size_var.get()[0])))
min_btn.pack(side='left', padx=5)

max_btn = ttk.Button(other_filters_frame, text="Filtr maksymalny",
                    command=lambda: apply_max_filter(int(size_var.get()[0])))
max_btn.pack(side='left', padx=5)

edge_frame = ttk.LabelFrame(main_frame, text="Detekcja krawędzi i wyostrzanie", padding="5")
edge_frame.pack(pady=10, fill='x')

sobel_frame = ttk.LabelFrame(edge_frame, text="a) Detekcja krawędzi Sobela", padding="5")
sobel_frame.pack(pady=5, fill='x', padx=5)

sobel_h_btn = ttk.Button(sobel_frame, text="Poziomy", command=apply_sobel_horizontal)
sobel_h_btn.pack(side='left', padx=5)

sobel_v_btn = ttk.Button(sobel_frame, text="Pionowy", command=apply_sobel_vertical)
sobel_v_btn.pack(side='left', padx=5)

sobel_d_btn = ttk.Button(sobel_frame, text="Ukośny", command=apply_sobel_diagonal)
sobel_d_btn.pack(side='left', padx=5)

laplacian_frame = ttk.LabelFrame(edge_frame, text="b) Wyostrzanie Laplace'a", padding="5")
laplacian_frame.pack(pady=5, fill='x', padx=5)

laplacian_btn = ttk.Button(laplacian_frame, text="Zastosuj Laplace'a", command=apply_laplacian)
laplacian_btn.pack(side='left', padx=5)

sharp_frame = ttk.LabelFrame(edge_frame, text="c) Maska wyostrzająca i wzmocnienie wysokich częstotliwości", padding="5")
sharp_frame.pack(pady=5, fill='x', padx=5)

unsharp_btn = ttk.Button(sharp_frame, text="Maska wyostrzająca", command=apply_unsharp_masking)
unsharp_btn.pack(side='left', padx=5)

highboost_btn = ttk.Button(sharp_frame, text="Wzmocnienie wysokich częstotliwości", command=apply_high_boost)
highboost_btn.pack(side='left', padx=5)

multi_step_frame = ttk.LabelFrame(main_frame, text="Wielostopniowe przetwarzanie", padding="5")
multi_step_frame.pack(pady=10, fill='x')
multi_step_btn = ttk.Button(multi_step_frame, 
                           text="Wielostopniowe przetwarzanie",
                           command=apply_multi_step_enhancement)
multi_step_btn.pack(padx=5, pady=5)
plt.ion()
row_scale.configure(command=lambda x: (update_profiles(), show_profile_lines()))
col_scale.configure(command=lambda x: (update_profiles(), show_profile_lines()))
ttk.Label(main_frame, text="Wybierz obraz:").pack(pady=5)
image_label = ttk.Label(image_window)
image_label.pack(pady=10)
image_selector = ttk.Combobox(main_frame, values=images, state="readonly")
image_selector.pack(pady=5)
image_selector.bind('<<ComboboxSelected>>', lambda e: (update_image(), update_profiles(), show_profile_lines()))
root.mainloop()
