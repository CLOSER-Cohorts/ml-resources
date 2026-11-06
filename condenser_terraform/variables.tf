# Put your input variables here
variable "namespace" {
  type        = string
  description = "Namespace that the VM will be deployed in"
}

variable "network_name" {
  type        = string
  description = "Name of a network in the namespace"
} 

variable "ssh_public_key_data" {
  type        = string
  description = "SSH public key data"
}
