with open("./keys/ed25519_private_key.pem", "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )


MODEL_NAME = "Logistic Regression for topic classification"
models_to_retrieve=[ 'uk.iser.ukhls',
    'uk.whitehall2',
    'uk.cls.nextsteps',
    'uk.lha',
    'uk.wchads',
    'uk.cls.bcs70',
    'uk.alspac']
ALIASES = [model.rsplit(".", 1)[-1] for model in models_to_retrieve]
for ALIAS in ALIASES:
    print(ALIAS)
    mv = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=ALIAS
    )
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri
    )
    model_hash = hash_directory(local_model_path)
    print(f"SHA-256: {model_hash}")
    signature = private_key.sign(
        bytes.fromhex(model_hash),
    )
    signature_b64 = base64.b64encode(signature).decode("ascii")
    mlflow_client.set_model_version_tag(
        name=MODEL_NAME,
        version=mv.version,
        key="model_sha256",
        value=model_hash
    )
    mlflow_client.set_model_version_tag(
        name=MODEL_NAME,
        version=mv.version,
        key="ed25519_signature",
        value=signature_b64
    )
    mlflow_client.set_model_version_tag(
        name=MODEL_NAME,
        version=mv.version,
        key="signature_algorithm",
        value="Ed25519"
    )