When logged into the n8n VM on condenser, you can backup and restore the n8n data with the following commands:

# Install software (if necessary)
sudo dnf install gnupg2 -y
sudo dnf install pinentry -y
sudo dnf install -y git
# Create the backup
tar -czvf n8n_backup.tar.gz -C /home/almalinux n8n_data
# Encrypt the backup
gpg --symmetric --cipher-algo AES256 n8n_backup.tar.gz
# Delete the unencrypted backup
rm n8n_backup.tar.gz
# Unencrypt and extract the data
gpg --output n8n_backup.tar.gz --decrypt n8n_backup.tar.gz.gpg
