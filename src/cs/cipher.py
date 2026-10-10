"""Full Image Encryption and Decryption Pipeline."""

import numpy as np
from PIL import Image

# Import internal modules
from cs.kem import generate_keypair, encapsulate, decapsulate
from cs.key_schedule import derive_chaotic_parameters
from cs.chaotic import iterate_6d_map
from cs.sbox import generate_dynamic_sbox
from cs.diffusion import (
    generate_diffusion_keystream,
    bidirectional_diffusion,
    inverse_bidirectional_diffusion
)

def encrypt_image(image_path: str, secret_key: bytes, output_path: str) -> None:
    # 1. Load Image
    img = Image.open(image_path)
    img_arr = np.array(img)
    original_shape = img_arr.shape
    
    # Handle Grayscale vs RGB/RGBA
    if len(original_shape) == 2:
        H, W = original_shape
        C = 1
        img_arr = img_arr.reshape(H, W, 1)
    else:
        H, W, C = original_shape

    num_pixels = H * W
    total_bytes = num_pixels * C

    # 2. Chaotic Map Setup (Length = H*W, Burn-in = 100)
    ivs, params = derive_chaotic_parameters(secret_key)
    _, chaotic_bytes = iterate_6d_map(ivs, params, length=num_pixels, burn_in=100)
    
    # 3. Phase 1: Channel-wise XOR Masking
    keystream_3d = np.empty_like(img_arr)
    for i in range(C):
        # Maps x1->R(0), x2->G(1), x3->B(2), x4->A(3)
        keystream_3d[:, :, i] = chaotic_bytes[:, i].reshape(H, W)
        
    xor_img_3d = np.bitwise_xor(img_arr, keystream_3d)
    
    # Flatten interleaved channels (R1, G1, B1, R2, G2, B2...)
    flat_img = xor_img_3d.flatten()
    
    # 4. Phase 2: Dynamic S-Box Substitution
    sbox, _ = generate_dynamic_sbox(secret_key[:8])
    sub_img = sbox[flat_img]
    
    # 5. Phase 3: Diffusion Layer
    kd = generate_diffusion_keystream(secret_key, total_bytes)
    diff_img = bidirectional_diffusion(sub_img, kd)
    
    # 6. Reconstruct and Save
    enc_img_arr = diff_img.reshape(original_shape)
    enc_img = Image.fromarray(enc_img_arr)
    enc_img.save(output_path, format="PNG")
    print(f"Encrypted image saved to: {output_path}")


def decrypt_image(encrypted_image_path: str, secret_key: bytes, output_path: str) -> None:
    # 1. Load Encrypted Image
    img = Image.open(encrypted_image_path)
    img_arr = np.array(img)
    original_shape = img_arr.shape
    
    if len(original_shape) == 2:
        H, W = original_shape
        C = 1
    else:
        H, W, C = original_shape
        
    num_pixels = H * W
    total_bytes = num_pixels * C
    flat_enc = img_arr.flatten()
    
    # 2. Phase 1: Inverse Diffusion
    kd = generate_diffusion_keystream(secret_key, total_bytes)
    un_diff_img = inverse_bidirectional_diffusion(flat_enc, kd)
    
    # 3. Phase 2: Inverse S-Box Substitution
    _, inv_sbox = generate_dynamic_sbox(secret_key[:8])
    un_sub_flat = inv_sbox[un_diff_img]
    un_sub_3d = un_sub_flat.reshape((H, W, C))
    
    # 4. Phase 3: Channel-wise Inverse XOR Masking
    ivs, params = derive_chaotic_parameters(secret_key)
    _, chaotic_bytes = iterate_6d_map(ivs, params, length=num_pixels, burn_in=100)
    
    keystream_3d = np.empty_like(un_sub_3d)
    for i in range(C):
        keystream_3d[:, :, i] = chaotic_bytes[:, i].reshape(H, W)
        
    dec_img_3d = np.bitwise_xor(un_sub_3d, keystream_3d)
    
    # 5. Reconstruct and Save
    if C == 1:
        dec_img_3d = dec_img_3d.reshape(H, W)
        
    dec_img = Image.fromarray(dec_img_3d)
    dec_img.save(output_path, format="PNG")
    print(f"Decrypted image saved to: {output_path}")


if __name__ == "__main__":
    # --- DEMONSTRATION PIPELINE ---
    
    # 1. ML-KEM Post-Quantum Key Exchange
    public_key, secret_key_bob = generate_keypair("ml-kem-768")
    shared_secret_alice, ciphertext = encapsulate(public_key, "ml-kem-768")
    shared_secret_bob = decapsulate(ciphertext, secret_key_bob, "ml-kem-768")
    assert shared_secret_alice == shared_secret_bob
    
    master_secret_key = shared_secret_alice
    
    # 2. Create Dummy RGB Image
    dummy_input_path = "test_original.png"
    encrypted_path = "test_encrypted.png"
    decrypted_path = "test_decrypted.png"
    
    random_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(random_rgb).save(dummy_input_path)
    
    # 3. Encrypt and Decrypt
    encrypt_image(dummy_input_path, master_secret_key, encrypted_path)
    decrypt_image(encrypted_path, master_secret_key, decrypted_path)
    
    # 4. Verification
    img_orig = np.array(Image.open(dummy_input_path))
    img_dec = np.array(Image.open(decrypted_path))
    
    assert np.array_equal(img_orig, img_dec), "Mismatch detected."
    print("Verification: Decrypted image perfectly matches the original input.")