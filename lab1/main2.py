import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# ==== GLOBAL VARIABLES ====
pictures_folder = r'C:\Users\wojte\Documents\LabyCpsio\lab1\obrazy'
images = [f for f in os.listdir(pictures_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
selected_image = None

# ==== FUNCTIONS ====
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
        
    # Get current row and column positions
    row = int(row_scale.get())
    col = int(col_scale.get())
    
    # Create a copy of the image to draw lines on
    display_image = Image.fromarray(selected_image)
    display_image = display_image.convert('RGB')  # Convert to RGB to draw colored lines
    
    # Draw horizontal line
    for x in range(display_image.width):
        display_image.putpixel((x, row), (255, 0, 0))  # Red line for horizontal profile
        
    # Draw vertical line    
    for y in range(display_image.height):
        display_image.putpixel((col, y), (0, 255, 0))  # Green line for vertical profile
    
    # Display the image with lines
    photo = ImageTk.PhotoImage(display_image)
    image_label.configure(image=photo)
    image_label.image = photo

def update_profiles(*args):
    if selected_image is None:
        return
        
    # Aktualizacja zakresu suwaków
    height, width = selected_image.shape
    row_scale.configure(to=height-1)
    col_scale.configure(to=width-1)
    
    # Profil poziomy
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
    
    # Profil pionowy
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
        # Convert to float32 for calculations
        img_float = selected_image.astype(np.float32)
        transformed = c * img_float
        
        # Normalize to 0-255 range
        if transformed.max() > 0:  # Avoid division by zero
            transformed = (transformed * 255 / transformed.max()).clip(0, 255)
        
        selected_image = transformed.astype(np.uint8)
        
        # Update the display
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        # Update profiles and lines
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
        # Convert to float32 for calculations
        img_float = selected_image.astype(np.float32)
        
        # Perform the transformation T(r) = c·log(1 + r)
        transformed = c * np.log1p(img_float)
        
        # Normalize to 0-255 range
        if transformed.max() > 0:  # Avoid division by zero
            transformed = (transformed * 255 / transformed.max()).clip(0, 255)
            
        selected_image = transformed.astype(np.uint8)
        
        # Update the display
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        # Update profiles and lines
        update_profiles()
        show_profile_lines()
    except ValueError:
        print("Please enter a valid number for constant c")

def contrast_transform():
    global selected_image
    if selected_image is None:
        return
        
    try:
        m = float(contrast_m_entry.get())  # m is now already normalized (0-1)
        e = float(contrast_e_entry.get())
        
        # Convert to float32 and normalize to 0-1 range
        img_float = selected_image.astype(np.float32) / 255.0
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # Calculate transformation in normalized space
        ratio = m / (img_float + epsilon)
        # Clip ratio to avoid extreme values
        ratio = np.clip(ratio, 0, 100)
        
        # Calculate result in normalized space then scale back to 0-255
        transformed = 1.0 / (1.0 + ratio ** e)
        transformed = (transformed * 255.0).clip(0, 255).astype(np.uint8)
        selected_image = transformed
        
        # Update the display
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        # Update profiles and lines
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
        
        # Convert to float32 and normalize to 0-1 range
        img_float = selected_image.astype(np.float32) / 255.0
        
        # Apply gamma correction: s = c * r^gamma
        transformed = c * (img_float ** gamma)
        
        # Scale back to 0-255 range
        transformed = (transformed * 255.0).clip(0, 255).astype(np.uint8)
        selected_image = transformed
        
        # Update the display
        display_image = Image.fromarray(selected_image)
        photo = ImageTk.PhotoImage(display_image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        # Update profiles and lines
        update_profiles()
        show_profile_lines()
        
    except ValueError:
        print("Please enter valid numbers for parameters c and gamma")

def histogram_equalization():
    global selected_image
    if selected_image is None:
        return
        
    # Calculate histogram and CDF
    hist, bins = np.histogram(selected_image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    # Normalize CDF to 0-255 range
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    
    # Create lookup table
    lookup_table = cdf_normalized.astype('uint8')
    
    # Store original image for comparison
    original_img = selected_image.copy()
    
    # Apply equalization
    selected_image = lookup_table[selected_image]
    
    # Update the display
    display_image = Image.fromarray(selected_image)
    photo = ImageTk.PhotoImage(display_image)
    image_label.configure(image=photo)
    image_label.image = photo
    
    # Update profiles and lines
    update_profiles()
    show_profile_lines()
    
    # Show histogram comparison
    plt.figure("Porównanie histogramów")
    plt.clf()
    
    # Original histogram
    plt.subplot(2, 1, 1)
    plt.hist(original_img.flatten(), bins=256, range=[0, 256], color='b', alpha=0.7)
    plt.title("Orginalny histogram")
    plt.xlabel("Poziom szarości")
    plt.ylabel("częstotliwość")
    plt.grid(True)
    
    # Equalized histogram
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
        
    # Create padded image
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    # Apply mean filter
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
        
    # Create padded image
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    # Apply median filter
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
        
    # Create padded image
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    # Apply min filter
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
        
    # Create padded image
    pad = size // 2
    padded = np.pad(selected_image, pad, mode='reflect')
    result = np.zeros_like(selected_image)
    
    # Apply max filter
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
        
    # Sobel horizontal mask
    mask = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
                    
    # Create padded image
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    # Apply filter
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    # Normalize to 0-255 range
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_sobel_vertical():
    global selected_image
    if selected_image is None:
        return
        
    # Sobel vertical mask
    mask = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
                    
    # Create padded image
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    # Apply filter
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    # Normalize to 0-255 range
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_sobel_diagonal():
    global selected_image
    if selected_image is None:
        return
        
    # Sobel diagonal mask (45 degrees)
    mask = np.array([[0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]])
                    
    # Create padded image
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    # Apply filter
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
            
    # Normalize to 0-255 range
    result = np.abs(result)
    result = (result * 255 / np.max(result)).clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_laplacian():
    global selected_image
    if selected_image is None:
        return
        
    # Laplacian mask
    mask = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
                    
    # Create padded image
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    # Apply filter
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            result[i, j] = np.sum(window * mask)
    
    # Add Laplacian to original for sharpening
    result = selected_image.astype(float) - result
    result = result.clip(0, 255).astype(np.uint8)
    selected_image = result
    update_display()

def apply_unsharp_masking():
    global selected_image
    if selected_image is None:
        return
        
    # Create blurred version using mean filter
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    blurred = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            blurred[i, j] = np.mean(window)
    
    # Calculate mask and apply unsharp masking
    mask = selected_image.astype(float) - blurred
    k = 2.0  # Sharpening strength
    result = selected_image.astype(float) + k * mask
    
    selected_image = result.clip(0, 255).astype(np.uint8)
    update_display()

def apply_high_boost():
    global selected_image
    if selected_image is None:
        return
        
    # Create blurred version using mean filter
    padded = np.pad(selected_image.astype(float), 1, mode='reflect')
    blurred = np.zeros_like(selected_image, dtype=float)
    
    for i in range(selected_image.shape[0]):
        for j in range(selected_image.shape[1]):
            window = padded[i:i+3, j:j+3]
            blurred[i, j] = np.mean(window)
    
    # High-boost filtering
    A = 1.5  # Boosting factor
    mask = selected_image.astype(float) - blurred
    result = A * selected_image.astype(float) + mask
    
    selected_image = result.clip(0, 255).astype(np.uint8)
    update_display()

def create_gaussian_kernel(size, sigma=1.0):
    # Create coordinates grid
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate Gaussian values
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    # Normalize the kernel
    return kernel / np.sum(kernel)

def apply_gaussian_filter(size):
    global selected_image
    if selected_image is None:
        return
        
    # Create Gaussian kernel
    sigma = size / 6.0  # Scale sigma with mask size
    kernel = create_gaussian_kernel(size, sigma)
    
    # Create padded image
    pad = size // 2
    padded = np.pad(selected_image.astype(float), pad, mode='reflect')
    result = np.zeros_like(selected_image, dtype=float)
    
    # Apply filter
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
        
    # Store original image
    original = selected_image.copy()
    
    # Create a new window for displaying steps
    steps_window = tk.Toplevel()
    steps_window.title("Multi-step Enhancement Process")
    steps_window.minsize(400, 300)
    
    # Create canvas with scrollbars
    canvas = tk.Canvas(steps_window)
    v_scrollbar = ttk.Scrollbar(steps_window, orient="vertical", command=canvas.yview)
    h_scrollbar = ttk.Scrollbar(steps_window, orient="horizontal", command=canvas.xview)
    
    # Create a frame inside canvas for content
    content_frame = ttk.Frame(canvas)
    
    # Configure scrolling
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    content_frame.bind("<Configure>", on_configure)
    
    # Bind mouse wheel
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Create window in canvas
    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    
    def show_step(img, title, row, col):
        # Convert to PhotoImage and display
        display_img = Image.fromarray(img)
        # Resize to reasonable size if too large
        if display_img.size[0] > 300:
            ratio = 300 / display_img.size[0]
            new_size = (300, int(display_img.size[1] * ratio))
            display_img = display_img.resize(new_size)
        photo = ImageTk.PhotoImage(display_img)
        label = ttk.Label(content_frame, image=photo)
        label.image = photo  # Keep reference
        label.grid(row=row, column=col, padx=5, pady=5)
        ttk.Label(content_frame, text=title).grid(row=row+1, column=col, padx=5, pady=(0,10))
    
    # Step a: Original image
    show_step(original, "a) Original Image", 0, 0)
    
    # Step b: Laplacian
    mask = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
    padded = np.pad(original.astype(float), 1, mode='reflect')
    laplacian = np.zeros_like(original, dtype=float)
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            window = padded[i:i+3, j:j+3]
            laplacian[i, j] = np.sum(window * mask)
    # Just use the Laplacian itself, scaled appropriately
    laplacian_result = np.abs(laplacian).clip(0, 255).astype(np.uint8)
    show_step(laplacian_result, "b) Laplacian", 0, 1)
    
    # Step c: Sum of original and Laplacian
    # Direct sum without division, then clip
    sum_result = (original.astype(float) + laplacian_result.astype(float)).clip(0, 255).astype(np.uint8)
    show_step(sum_result, "c) Sum a + b", 0, 2)
    
    # Step d: Sobel gradient on the sum result
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    gradient_x = np.zeros_like(original, dtype=float)
    gradient_y = np.zeros_like(original, dtype=float)
    
    # Apply Sobel on the sum_result
    padded = np.pad(sum_result.astype(float), 1, mode='reflect')
    
    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            window = padded[i:i+3, j:j+3]
            gradient_x[i, j] = np.abs(np.sum(window * sobel_x))
            gradient_y[i, j] = np.abs(np.sum(window * sobel_y))
    
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    # Adjust normalization to get brighter edges
    gradient = (gradient * 255 / gradient.max()).clip(0, 255).astype(np.uint8)
    # Apply threshold to make edges more pronounced
    gradient[gradient < 30] = 0  # Remove weak edges
    show_step(gradient, "d) Sobel Gradient", 2, 0)
    
    # Step e: Mean filter (5x5) applied to gradient result
    mean_result = np.zeros_like(gradient, dtype=float)
    padded_grad = np.pad(gradient.astype(float), 2, mode='reflect')
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            window = padded_grad[i:i+5, j:j+5]
            mean_result[i, j] = np.mean(window)
    mean_result = mean_result.clip(0, 255).astype(np.uint8)
    show_step(mean_result, "e) Mean Filter 5x5", 2, 1)
    
    # Step f: Final enhancement (gamma correction with adjusted parameters)
    gamma = 0.6  # Slightly higher gamma to match example
    c = 1.2     # Slightly higher contrast
    final_result = c * (mean_result.astype(float) / 255.0) ** gamma
    final_result = (final_result * 255).clip(0, 255).astype(np.uint8)
    show_step(final_result, "f) Gamma Correction", 2, 2)
    
    # Update the main display with final result
    selected_image = final_result
    update_display()
    
    # Pack scrollbars and canvas
    h_scrollbar.pack(side="bottom", fill="x")
    v_scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    
    # Configure canvas scrolling
    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    # Add a close button at the bottom
    close_btn = ttk.Button(steps_window, text="Close", command=steps_window.destroy)
    close_btn.pack(side="bottom", pady=5)


# ==== GUI ====
root = tk.Tk()
root.title("Przetwarzanie obrazów")

def on_closing():
    plt.close('all')  # Close all matplotlib windows
    image_window.destroy()  # Close image window
    root.quit()  # Stop the mainloop
    root.destroy()  # Destroy the main window

root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a canvas with scrollbar
canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

# Configure the canvas
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# Bind mouse wheel to scrolling
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
canvas.bind_all("<MouseWheel>", _on_mousewheel)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Set minimum window size and canvas width
root.minsize(600, 400)
canvas.configure(width=580)  # Set fixed width to prevent horizontal scrolling

# Pack the scrollbar and canvas
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Main frame inside scrollable frame
main_frame = ttk.Frame(scrollable_frame, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# okno z obrazem
image_window = tk.Toplevel()
image_window.title("Podgląd obrazu")

# Remove the empty plot windows creation
# Okna z wykresami
# profile_window_h = tk.Toplevel()
# profile_window_h.title("Profil poziomy")
# profile_window_v = tk.Toplevel()
# profile_window_v.title("Profil pionowy")

# Suwaki do wyboru linii
ttk.Label(main_frame, text="Wybierz wiersz:").pack(pady=5)
row_scale = tk.Scale(main_frame, from_=0, to=500, orient='horizontal')
row_scale.pack(fill='x', padx=10)

ttk.Label(main_frame, text="Wybierz kolumnę:").pack(pady=5) 
col_scale = tk.Scale(main_frame, from_=0, to=500, orient='horizontal')
col_scale.pack(fill='x', padx=10)

# Remove the old transform frame and add new sections
transform_frame = ttk.LabelFrame(main_frame, text="Point Transformations", padding="5")
transform_frame.pack(pady=10, fill='x')

# Multiplication transform
multiply_frame = ttk.LabelFrame(transform_frame, text="a) Multiply by constant", padding="5")
multiply_frame.pack(pady=5, fill='x', padx=5)

ttk.Label(multiply_frame, text="c:").pack(side='left', padx=5)
multiply_c_entry = ttk.Entry(multiply_frame, width=8)
multiply_c_entry.insert(0, "1.0")
multiply_c_entry.pack(side='left', padx=5)

multiply_btn = ttk.Button(multiply_frame, text="Apply T(r) = c·r", command=multiply_by_constant)
multiply_btn.pack(side='left', padx=5)

# Logarithmic transform
log_frame = ttk.LabelFrame(transform_frame, text="b) Logarithmic transform", padding="5")
log_frame.pack(pady=5, fill='x', padx=5)

ttk.Label(log_frame, text="c:").pack(side='left', padx=5)
log_c_entry = ttk.Entry(log_frame, width=8)
log_c_entry.insert(0, "1.0")
log_c_entry.pack(side='left', padx=5)

log_btn = ttk.Button(log_frame, text="Apply T(r) = c·log(1+r)", command=logarithmic_transform)
log_btn.pack(side='left', padx=5)

# Contrast transform
contrast_frame = ttk.LabelFrame(transform_frame, text="c) Contrast transformation", padding="5")
contrast_frame.pack(pady=5, fill='x', padx=5)

param_frame = ttk.Frame(contrast_frame)
param_frame.pack(side='left', padx=5)

ttk.Label(param_frame, text="m:").grid(row=0, column=0, padx=5)
contrast_m_entry = ttk.Entry(param_frame, width=8)
contrast_m_entry.insert(0, "0.5")  # Normalized value
contrast_m_entry.grid(row=0, column=1, padx=5)

ttk.Label(param_frame, text="e:").grid(row=0, column=2, padx=5)
contrast_e_entry = ttk.Entry(param_frame, width=8)
contrast_e_entry.insert(0, "4")
contrast_e_entry.grid(row=0, column=3, padx=5)

contrast_btn = ttk.Button(contrast_frame, text="Apply T(r) = 255/(1+(m/r)^e)", command=contrast_transform)
contrast_btn.pack(side='left', padx=5)

# Gamma correction
gamma_frame = ttk.LabelFrame(transform_frame, text="d) Gamma correction", padding="5")
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

gamma_btn = ttk.Button(gamma_frame, text="Apply s = c·r^γ", command=gamma_correction)
gamma_btn.pack(side='left', padx=5)

# Histogram Equalization
hist_eq_frame = ttk.LabelFrame(transform_frame, text="e) Histogram Equalization", padding="5")
hist_eq_frame.pack(pady=5, fill='x', padx=5)

hist_eq_btn = ttk.Button(hist_eq_frame, text="Apply Histogram Equalization", command=histogram_equalization)
hist_eq_btn.pack(padx=5, pady=5)

# Modify the filters section to group low-pass filters together
filters_frame = ttk.LabelFrame(main_frame, text="Filters", padding="5")
filters_frame.pack(pady=10, fill='x')

# Low-pass filters section
lowpass_frame = ttk.LabelFrame(filters_frame, text="Low-pass Filters", padding="5")
lowpass_frame.pack(pady=5, fill='x', padx=5)

# Size selection with more options
filter_sizes = ['3x3', '5x5', '7x7', '9x9', '11x11']
size_var = tk.StringVar(value='3x3')
ttk.Label(lowpass_frame, text="Mask size:").pack(side='left', padx=5)
size_combo = ttk.Combobox(lowpass_frame, values=filter_sizes, textvariable=size_var, state='readonly', width=6)
size_combo.pack(side='left', padx=5)

# Mean and Gaussian filter buttons
mean_btn = ttk.Button(lowpass_frame, text="Mean Filter",
                     command=lambda: apply_mean_filter(int(size_var.get()[0])))
mean_btn.pack(side='left', padx=5)

gaussian_btn = ttk.Button(lowpass_frame, text="Gaussian Filter",
                         command=lambda: apply_gaussian_filter(int(size_var.get()[0])))
gaussian_btn.pack(side='left', padx=5)

# Other filters section (median, min, max)
other_filters_frame = ttk.LabelFrame(filters_frame, text="Other Filters", padding="5")
other_filters_frame.pack(pady=5, fill='x', padx=5)

median_btn = ttk.Button(other_filters_frame, text="Median Filter",
                       command=lambda: apply_median_filter(int(size_var.get()[0])))
median_btn.pack(side='left', padx=5)

min_btn = ttk.Button(other_filters_frame, text="Min Filter",
                    command=lambda: apply_min_filter(int(size_var.get()[0])))
min_btn.pack(side='left', padx=5)

max_btn = ttk.Button(other_filters_frame, text="Max Filter",
                    command=lambda: apply_max_filter(int(size_var.get()[0])))
max_btn.pack(side='left', padx=5)

# Edge Detection and Sharpening
edge_frame = ttk.LabelFrame(main_frame, text="Edge Detection and Sharpening", padding="5")
edge_frame.pack(pady=10, fill='x')

# Sobel operators
sobel_frame = ttk.LabelFrame(edge_frame, text="a) Sobel Edge Detection", padding="5")
sobel_frame.pack(pady=5, fill='x', padx=5)

sobel_h_btn = ttk.Button(sobel_frame, text="Horizontal", command=apply_sobel_horizontal)
sobel_h_btn.pack(side='left', padx=5)

sobel_v_btn = ttk.Button(sobel_frame, text="Vertical", command=apply_sobel_vertical)
sobel_v_btn.pack(side='left', padx=5)

sobel_d_btn = ttk.Button(sobel_frame, text="Diagonal", command=apply_sobel_diagonal)
sobel_d_btn.pack(side='left', padx=5)

# Laplacian sharpening
laplacian_frame = ttk.LabelFrame(edge_frame, text="b) Laplacian Sharpening", padding="5")
laplacian_frame.pack(pady=5, fill='x', padx=5)

laplacian_btn = ttk.Button(laplacian_frame, text="Apply Laplacian", command=apply_laplacian)
laplacian_btn.pack(side='left', padx=5)

# Unsharp masking and high-boost
sharp_frame = ttk.LabelFrame(edge_frame, text="c) Unsharp Masking and High-boost", padding="5")
sharp_frame.pack(pady=5, fill='x', padx=5)

unsharp_btn = ttk.Button(sharp_frame, text="Unsharp Masking", command=apply_unsharp_masking)
unsharp_btn.pack(side='left', padx=5)

highboost_btn = ttk.Button(sharp_frame, text="High-boost", command=apply_high_boost)
highboost_btn.pack(side='left', padx=5)

# Multi-step enhancement section
multi_step_frame = ttk.LabelFrame(main_frame, text="Multi-step Enhancement Process", padding="5")
multi_step_frame.pack(pady=10, fill='x')

multi_step_btn = ttk.Button(multi_step_frame, 
                           text="Apply Multi-step Enhancement (Show All Steps)",
                           command=apply_multi_step_enhancement)
multi_step_btn.pack(padx=5, pady=5)

# Włączenie interaktywnego trybu matplotlib
plt.ion()

# Powiązanie aktualizacji wykresów
row_scale.configure(command=lambda x: (update_profiles(), show_profile_lines()))
col_scale.configure(command=lambda x: (update_profiles(), show_profile_lines()))




# wybór obrazu
ttk.Label(main_frame, text="Select image:").pack(pady=5)
image_label = ttk.Label(image_window)
image_label.pack(pady=10)



image_selector = ttk.Combobox(main_frame, values=images, state="readonly")
image_selector.pack(pady=5)
image_selector.bind('<<ComboboxSelected>>', lambda e: (update_image(), update_profiles(), show_profile_lines()))






# Uruchomienie głównej pętli
root.mainloop()
