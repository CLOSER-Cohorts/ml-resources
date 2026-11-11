When logged into the n8n VM on condenser, you can backup and restore the n8n data with the following commands:

# Install software (if necessary)
```
sudo dnf install gnupg2 -y
sudo dnf install pinentry -y
sudo dnf install -y git
```
You may need to define your git account
```
git config --global user.email "you@example.com"
```
You may also need to enable ssh authentication, copy the ssh key to your github 
account (not shown in this README) and then set the remote url before using git
to push to the repository
```
ssh-keygen -t ed25519 -C "n8n-backup"
cat ~/.ssh/id_ed25519.pub
git remote set-url origin git@github.com:CLOSER-Cohorts/ml-resources.git
```
# Create/encrypt/decrypt the backup
```
tar -czvf n8n_backup.tar.gz -C /home/almalinux n8n_data
gpg --symmetric --cipher-algo AES256 n8n_backup.tar.gz
rm n8n_backup.tar.gz
gpg --output n8n_backup.tar.gz --decrypt n8n_backup.tar.gz.gpg
```
