from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives import serialization
import hashlib
from pathlib import Path


def create_keys():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open("./keys/ed25519_private_key.pem", "wb") as f:
        f.write(private_bytes)
    with open("./keys/ed25519_public_key.pem", "wb") as f:
        f.write(public_bytes)
    
def get_keys():
    with open("./keys/ed25519_private_key.pem", "rb") as f:
        private_key = serialization.load_pem_private_key(
        f.read(),
        password=None
    )
    with open("./keys/ed25519_public_key.pem", "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())
    return {"private_key": private_key,
            "public_key": public_key}

def hash_directory(directory: str) -> str:
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(directory).as_posix()
        # Include filename
        digest.update(relative_path.encode("utf-8"))
        # Include file contents
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()