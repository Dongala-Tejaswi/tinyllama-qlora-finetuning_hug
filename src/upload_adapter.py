from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="TejaswiDongala/tinyllama-qlora",
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path="../adapter",
    repo_id="TejaswiDongala/tinyllama-qlora",
    repo_type="model"
)

print("Upload Complete!")