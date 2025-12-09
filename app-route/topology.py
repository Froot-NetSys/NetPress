import os
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node, RemoteController
from mininet.cli import CLI
from mininet.log import info, setLogLevel
from mininet.node import OVSController
import socket
import fcntl


class LinuxRouter(Node):
    """A Node with IP forwarding enabled."""

    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        # Enable IP forwarding on the router
        self.cmd('sysctl -w net.ipv4.ip_forward=1')

    def terminate(self):
        self.cmd('sysctl -w net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()


class NetworkTopo(Topo):
    """A LinuxRouter connecting multiple IP subnets"""

    def build(self, num_hosts_per_subnet=5, num_switches=5, subnets=None, prefix=""):
        # Create the router with a unique name
        router_name = f"{prefix}r0"
        router = self.addNode(router_name, cls=LinuxRouter, ip=subnets[0][0])

        # Create switches
        switches = [self.addSwitch(f"{prefix}s{i+1}") for i in range(num_switches)]

        # Link each switch to the router with a unique interface
        for i, switch in enumerate(switches):
            self.addLink(switch, router, intfName2=f"{router_name}-eth{i+1}", 
                         params2={'ip': subnets[i % len(subnets)][0]})

            # Create multiple hosts for each switch
            for j in range(num_hosts_per_subnet):
                host_ip = f'{subnets[i][0].split(".")[0]}.{subnets[i][0].split(".")[1]}.{subnets[i][0].split(".")[2]}.{100+j}/24'
                host_name = f"{prefix}h{i*num_hosts_per_subnet + j + 1}"
                self.addHost(host_name, ip=host_ip, defaultRoute=f'via {subnets[i][0].split("/")[0]}')
                self.addLink(host_name, switch)


def generate_subnets(num_switches, base_ip=[192, 168, 1, 1]):
    """Generate subnets, ensuring the third octet does not exceed 255"""
    subnets = []
    
    for i in range(num_switches):
        subnet_ip = base_ip.copy()
        subnet_ip[2] = (subnet_ip[2] + i) % 255  # Avoid exceeding 255

        subnet = f"{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.{subnet_ip[3]}/24"
        host_ip = f"{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.100/24"
        subnet_address = f"{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.0/24"
        
        subnets.append((subnet, host_ip, subnet_address))
    
    return subnets


def initialize_network(num_hosts_per_subnet, num_switches, unique_id):
    import sys
    print(f"  [DEBUG] Generating subnets...")
    sys.stdout.flush()
    subnets = generate_subnets(num_switches)
    prefix = f"p{unique_id % 100}_"

    print(f"  [DEBUG] Building topology...")
    sys.stdout.flush()
    # Build the topology
    topo = NetworkTopo(num_hosts_per_subnet, num_switches, subnets, prefix)

    print(f"  [DEBUG] Creating Mininet instance...")
    sys.stdout.flush()
    # Explicitly disable the default controller
    # Note: waitConnected=False to avoid hanging in Docker
    net = Mininet(topo=topo, waitConnected=False, controller=None, cleanup=True, autoSetMacs=True)

    print(f"  [DEBUG] Adding controller on port {6700 + unique_id % 100}...")
    sys.stdout.flush()
    # Use a unique port for the controller
    controller_port = 6700 + unique_id % 100  # Ensure this port is unique
    controller = net.addController(name=f"c{unique_id}", controller=OVSController, port=controller_port)

    print(f"  [DEBUG] Starting network...")
    sys.stdout.flush()
    net.start()

    print(f"  [DEBUG] Enabling IP forwarding on router...")
    sys.stdout.flush()
    # Enable IP forwarding on the router
    router = net.get(f"{prefix}r0")
    info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))

    print(f"✅ Process {unique_id}: Mininet started with OVSController on port {controller_port}")
    sys.stdout.flush()
    return subnets, topo, net, router

