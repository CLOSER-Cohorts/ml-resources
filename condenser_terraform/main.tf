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
    runcmd:
      - - systemctl
        - enable
        - --now
        - qemu-guest-agent.service
      - sed -i '/^Listen 80$/a Listen 8000' /etc/httpd/conf/httpd.conf
      - setsebool -P httpd_can_network_connect on
      - dnf install -y git
      - dnf install -y python3.12 python3.12-pip
      - dnf install -y httpd
      - cd /home/almalinux
      - python3.12 -m venv .venv312
      - source .venv312/bin/activate
      - python -m ensurepip --upgrade
      - python -m pip install --upgrade pip
      - curl -fsSL https://raw.githubusercontent.com/CLOSER-Cohorts/ml-resources/refs/heads/main/condenser_terraform/mlapps-proxy.conf -o /etc/httpd/conf.d/mlapps-proxy.conf
      - chown root:root /etc/httpd/conf.d/mlapps-proxy.conf
      - chmod 644 /etc/httpd/conf.d/mlapps-proxy.conf
      - git clone https://github.com/CLOSER-Cohorts/ml-resources
      - cd ml-resources/ai_apprenticeship_oliver
      - python -m pip install -e .
      - "nohup mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts 127.0.0.1:5000,localhost:5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination file:./mlruns &"
      - "until curl -sf http://127.0.0.1:5000/health; do echo 'Start MLflow:'; sleep 2; done"
      - "nohup python -m uvicorn projects.am1_project.api.main:app --host 127.0.0.1 --port 8001 &"
      - semanage port -a -t http_port_t -p tcp 8000
      - "until curl -sf http://127.0.0.1:8001; do echo 'Start FastAPI:'; sleep 2; done"
      - "nohup streamlit run ./projects/am1_project/ui/am1_ui_app.py &"
      - - systemctl
        - enable
        - --now
        - httpd
    ssh_authorized_keys:
      - ${var.ssh_public_key_data}
    EOF
  }

  labels = {
  "condenser.ingress/isEnabled"     = true
  "condenser.ingress.mlapps/hostname" = "mlapps"
  "condenser.ingress.fastapi/hostname" = "fastapi"
  "condenser.ingress.fastapi/port" = "8000"
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
      - mkdir -p /home/almalinux/n8n_data
      - chown -R 1000:1000 /home/almalinux/n8n_data
      - dnf -y install zstd
      - curl -fsSL https://ollama.com/install.sh | sh
      - semanage port -a -t http_port_t -p tcp 11434
      - mkdir -p /etc/systemd/system/ollama.service.d
      - |   
        cat > /etc/systemd/system/ollama.service.d/override.conf <<'EOT'
        [Service]
        Environment="HOME=/usr/share/ollama"
        Environment="OLLAMA_HOST=0.0.0.0:11434"
        EOT
      - systemctl daemon-reload
      - systemctl restart ollama
      - until curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; do sleep 2; done
      - sudo -u ollama env HOME=/usr/share/ollama OLLAMA_HOST=127.0.0.1:11434 /usr/local/bin/ollama pull qwen2.5:7b
      # Add Docker's official repository
      - sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
      - sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
      - sudo echo "RUN DOCKER"
      - sudo systemctl enable --now docker
      - sudo echo "START CONTAINER"
      - sudo docker run -d --restart unless-stopped --name n8n  -p 80:5678  -e GENERIC_TIMEZONE="Europe/London" -e TZ="Europe/London" -e N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true -e N8N_RUNNERS_ENABLED=true -e N8N_EDITOR_BASE_URL="https://n8n.sri-ml.condenser.arc.ucl.ac.uk/" -e N8N_WEBHOOK_URL="https://n8n.sri-ml.condenser.arc.ucl.ac.uk/" -e N8N_PROXY_HOPS=1 --mount type=bind,source=/home/almalinux/n8n_data,target=/home/node/.n8n n8nio/n8n
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
  "condenser.ingress.ollama/hostname" = "ollama"
  "condenser.ingress.ollama/port" = "11434" 
}
}
