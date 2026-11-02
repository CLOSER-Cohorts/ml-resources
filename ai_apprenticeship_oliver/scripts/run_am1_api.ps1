cd C:\Users\qtnzoly\Development\fastApi\ml-resources

..\.venv\Scripts\Activate.ps1

git pull origin main

pip install -r requirements.txt

python -m uvicorn projects.am1_project.api.main:app