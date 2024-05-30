import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  
    image_array = np.array(image)
    return image_array

image_path = '/Users/yigitbickici/Documents/GitHub/Unternehmenssoftware-Project/IMG_7864.png'
image_matrix = load_image(image_path)
plt.imshow(image_matrix, cmap='gray')
plt.title('Original Image')
plt.show()


def compress_image(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    compressed_image = np.dot(U_k, np.dot(S_k, Vt_k))
    return compressed_image

k = 50  
compressed_image_matrix = compress_image(image_matrix, k)
plt.imshow(compressed_image_matrix, cmap='gray')
plt.title(f'Compressed image with the rank of {k}')
plt.show()


def save_image(image_matrix, output_path):
    image = Image.fromarray(np.uint8(image_matrix))
    image.save(output_path)


k_values = [5, 20, 50, 100]
for k in k_values:
    compressed_image_matrix = compress_image(image_matrix, k)
    output_path = f'compressed_image{k}.jpg'
    save_image(compressed_image_matrix, output_path)
    plt.imshow(compressed_image_matrix, cmap='gray')
    plt.title(f'Compressed image with the rank of {k}')
    plt.show()




