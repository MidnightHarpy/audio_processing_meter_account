from functools import lru_cache
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import base64


class DeterministicCipher:
    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            raise ValueError("ENCRYPTION_KEY не найден в окружении")

        salt = b'fixed_salt_value'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        self.key = kdf.derive(key.encode())
        self.backend = default_backend()

    @lru_cache(maxsize=1000)
    def encrypt(self, data: str) -> str:
        digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        digest.update(data.encode())
        iv = digest.finalize()[:16]

        cipher = Cipher(algorithms.AES(self.key), modes.CTR(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(data.encode()) + encryptor.finalize()
        return base64.urlsafe_b64encode(iv + encrypted).decode()

    @lru_cache(maxsize=1000)
    def decrypt(self, encrypted_str: str) -> str:
        try:
            data = base64.urlsafe_b64decode(encrypted_str.encode())
            iv, encrypted = data[:16], data[16:]
            cipher = Cipher(algorithms.AES(self.key), modes.CTR(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            return (decryptor.update(encrypted) + decryptor.finalize()).decode()
        except Exception as e:
            raise ValueError(f"Ошибка дешифрования: {str(e)}")


cipher = DeterministicCipher()