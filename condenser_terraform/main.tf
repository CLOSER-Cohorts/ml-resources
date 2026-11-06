# This is the main entry point for your Terraform code.

data "harvester_image" "img" {
  display_name = "almalinux-9.4-20240507"
  namespace    = "harvester-public"
}

resource "harvester_virtualmachine" "vm" {
  name        = "mlapps"
  namespace   = var.namespace
  description = "ML VM"
  hostname    = "mlapps"

  cpu    = 1
  memory = "8Gi"

  restart_after_update = true
  efi                  = true
  secure_boot          = true
  run_strategy         = "RerunOnFailure"
  reserved_memory      = "100Mi"
  machine_type         = "q35"

  disk {
    name       = "rootdisk"
    type       = "disk"
    size       = "100Gi"
    bus        = "virtio"
    boot_order = 1

    image       = data.harvester_image.img.id
    auto_delete = true
  }

  network_interface {
    name           = "nic-1"
    wait_for_lease = true
    type           = "bridge"
    network_name   = var.network_name
  }

  cloudinit {
    user_data = <<EOF
    #cloud-config
    package_update: true
    package_upgrade: true
    packages:
      - qemu-guest-agent
      - httpd
    write_files:
    - path: /etc/httpd/conf.d/streamlit-proxy.conf
      owner: root:root
      permissions: '0644'
      content: |
        <VirtualHost *:80>
            ServerName localhost
            ProxyRequests Off
            ProxyPreserveHost On
            ProxyPass /_stcore/stream ws://127.0.0.1:8501/_stcore/stream
            ProxyPassReverse /_stcore/stream ws://127.0.0.1:8501/_stcore/stream
            ProxyPass / http://127.0.0.1:8501/
            ProxyPassReverse / http://127.0.0.1:8501/
        </VirtualHost>
    runcmd:
      - - systemctl
        - enable
        - --now
        - qemu-guest-agent.service
      - - systemctl
        - enable
        - --now
          - httpd
      - setsebool -P httpd_can_network_connect on
      - dnf install -y git
      - dnf install -y python3.12 python3.12-pip
      - cd /home/almalinux
      - python3.12 -m venv .venv312
      - source .venv312/bin/activate
      - git clone https://github.com/CLOSER-Cohorts/ml-resources
      - cd ml-resources/ai_apprenticeship_oliver
      - python -m pip install -e .
      - mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts 127.0.0.1:5000,localhost:5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination file:./mlruns &
      - python -m uvicorn projects.am1_project.api.main:app &
      - streamlit run ./projects/am1_project/ui/am1_ui_app.py &
    ssh_authorized_keys:
      - ${var.ssh_public_key_data}
    EOF
  }

  labels = {
  "condenser.ingress/isEnabled"     = true
  "condenser.ingress.mlapps/hostname" = "mlapps"
}
}

resource "harvester_virtualmachine" "n8n_vm" {
  name        = "n8n"
  namespace   = var.namespace
  description = "n8n VM"
  hostname    = "n8n"

  cpu    = 1
  memory = "8Gi"

  restart_after_update = true
  efi                  = true
  secure_boot          = true
  run_strategy         = "RerunOnFailure"
  reserved_memory      = "100Mi"
  machine_type         = "q35"

  disk {
    name       = "rootdisk"
    type       = "disk"
    size       = "100Gi"
    bus        = "virtio"
    boot_order = 1

    image       = data.harvester_image.img.id
    auto_delete = true
  }

  network_interface {
    name           = "nic-1"
    wait_for_lease = true
    type           = "bridge"
    network_name   = var.network_name
  }

  cloudinit {
    user_data = <<EOF
    #cloud-config
    package_update: true
    package_upgrade: true
    packages:
      - qemu-guest-agent
      - dnf-plugins-core
    runcmd:
      # Add Docker's official repository
      - sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
      # Install Docker Engine and the Docker CLI
      - sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      - sudo echo "RUN DOCKER"
      # Enable and start Docker
      - sudo systemctl enable --now docker
      # Start the n8n container
      - sudo echo "START CONTAINER"
      - sudo docker run -d --restart unless-stopped --name n8n  -p 80:5678  -e GENERIC_TIMEZONE="Europe/London" -e TZ="Europe/London" -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true -e N8N_RUNNERS_ENABLED=true -v n8n_data:/home/almalinux/n8n_data/.n8n  n8nio/n8n
      - - systemctl
        - enable
        - --now
        - qemu-guest-agent.service
    ssh_authorized_keys:
      - ${var.ssh_public_key_data}
    EOF
  }

  labels = {
  "condenser.ingress/isEnabled"     = true
  "condenser.ingress.n8n/hostname" = "n8n"
}
}
