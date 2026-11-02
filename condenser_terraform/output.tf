# List your outputs here.

output "ip_address" {
  value = harvester_virtualmachine.n8n_vm.network_interface[0].ip_address
}
