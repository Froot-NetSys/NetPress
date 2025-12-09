from mininet.log import setLogLevel, info, lg
import random
import json
from itertools import combinations

def error_disable_routing(router, subnets):
    # Multiple ways to disable routing
    methods = [
        # Basic sysctl disable
        # FIX: sysctl -w net.ipv4.ip_forward=1
        lambda: router.cmd('sysctl -w net.ipv4.ip_forward=0'),
        
        # Block all forwarded packets with iptables
        # FIX: iptables -P FORWARD ACCEPT
        lambda: router.cmd('iptables -P FORWARD DROP'),
        
        # Set routing tables to reject all forwarding
        # FIX: ip rule del prohibit all pref 0
        lambda: router.cmd('ip rule add prohibit all pref 0'),
        
        # Partial routing disable (only some subnets)
        # FIX: iptables -D FORWARD -s SUBNET -j DROP
        lambda: router.cmd(f'iptables -A FORWARD -s {random.choice(subnets)[2]} -j DROP')
    ]
    chosen_method = random.choice(methods)
    info(f'*** Injecting error: Disabling IP forwarding using method {methods.index(chosen_method)+1}\n')
    chosen_method()


def error_disable_interface(router, subnets):
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_disable = random.choice(interfaces)
    
    methods = [
        # Completely disable interface
        # FIX: ifconfig INTERFACE up
        lambda: router.cmd(f'ifconfig {interface_to_disable} down'),
        
        # Use ip command instead
        # FIX: ip link set INTERFACE up
        lambda: router.cmd(f'ip link set {interface_to_disable} down'),
        
        # Set extremely low MTU
        # FIX: ip link set INTERFACE mtu 1500
        lambda: router.cmd(f'ip link set {interface_to_disable} mtu 68'),
        
        # Set interface to promiscuous mode
        # FIX: ip link set INTERFACE promisc off
        lambda: router.cmd(f'ip link set {interface_to_disable} promisc on'),
        
        # Add artificial packet loss
        # FIX: tc qdisc del dev INTERFACE root
        lambda: router.cmd(f'tc qdisc add dev {interface_to_disable} root netem loss {random.randint(50, 95)}%')
    ]
    chosen_method = random.choice(methods)
    info(f'*** Injecting error: Interfering with interface {interface_to_disable} using method {methods.index(chosen_method)+1}\n')
    chosen_method()


def error_remove_ip(router, subnets):
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_modify = random.choice(interfaces)
    interface_index = int(interface_to_modify.split('eth')[1]) - 1
    current_subnet = subnets[interface_index % len(subnets)][0].split('/')[0]
    
    methods = [
        # Completely remove IP
        # FIX: ip addr add ORIGINAL_IP/24 dev INTERFACE
        lambda: router.cmd(f'ip addr flush dev {interface_to_modify}'),
        
        # Set incorrect IP (different subnet)
        # FIX: ip addr flush dev INTERFACE && ip addr add ORIGINAL_IP/24 dev INTERFACE
        lambda: router.cmd(f'ip addr flush dev {interface_to_modify} && ip addr add 10.{random.randint(1,254)}.{random.randint(1,254)}.1/24 dev {interface_to_modify}'),
        
        # Set incorrect subnet mask
        # FIX: ip addr flush dev INTERFACE && ip addr add ORIGINAL_IP/24 dev INTERFACE
        lambda: router.cmd(f'ip addr flush dev {interface_to_modify} && ip addr add {current_subnet}/{random.choice([8, 16, 30, 31, 32])} dev {interface_to_modify}'),
        
        # Set duplicate IP from another subnet
        # FIX: ip addr flush dev INTERFACE && ip addr add ORIGINAL_IP/24 dev INTERFACE
        lambda: router.cmd(f'ip addr flush dev {interface_to_modify} && ip addr add {random.choice([s[0] for s in subnets if s[0].split("/")[0] != current_subnet])} dev {interface_to_modify}')
    ]
    chosen_method = random.choice(methods)
    info(f'*** Injecting error: Modifying IP on interface {interface_to_modify} using method {methods.index(chosen_method)+1}\n')
    chosen_method()


def error_drop_traffic_to_from_subnet(router, subnets):
    subnet_to_drop = random.choice(subnets)
    subnet_address = subnet_to_drop[2]
    
    methods = [
        # Simple drop
        # FIX: iptables -D INPUT -s SUBNET -j DROP && iptables -D OUTPUT -d SUBNET -j DROP
        lambda: [router.cmd(f'iptables -A INPUT -s {subnet_address} -j DROP'), 
                router.cmd(f'iptables -A OUTPUT -d {subnet_address} -j DROP')],
        
        # Reject instead of drop (generates ICMP errors)
        # FIX: iptables -D INPUT -s SUBNET -j REJECT && iptables -D OUTPUT -d SUBNET -j REJECT
        lambda: [router.cmd(f'iptables -A INPUT -s {subnet_address} -j REJECT'), 
                router.cmd(f'iptables -A OUTPUT -d {subnet_address} -j REJECT')],
        
        # Only drop TCP traffic
        # FIX: iptables -D INPUT -s SUBNET -p tcp -j DROP && iptables -D OUTPUT -d SUBNET -p tcp -j DROP
        lambda: [router.cmd(f'iptables -A INPUT -s {subnet_address} -p tcp -j DROP'), 
                router.cmd(f'iptables -A OUTPUT -d {subnet_address} -p tcp -j DROP')],
        
        # Add high latency
        # FIX: tc qdisc del dev INTERFACE root
        lambda: router.cmd(f'tc qdisc add dev r0-eth{subnets.index(subnet_to_drop)+1} root netem delay {random.randint(1000, 5000)}ms'),
        
        # Corrupt packets
        # FIX: tc qdisc del dev INTERFACE root
        lambda: router.cmd(f'tc qdisc add dev r0-eth{subnets.index(subnet_to_drop)+1} root netem corrupt {random.randint(20, 50)}%')
    ]
    chosen_method = random.choice(methods)
    info(f'*** Injecting error: Traffic manipulation for subnet {subnet_address} using method {methods.index(chosen_method)+1}\n')
    chosen_method()


def error_wrong_routing_table(router, subnets):
    num_subnets = len(subnets)
    selected_indices = random.sample(range(num_subnets), 2)
    
    interface1 = selected_indices[0] + 1
    interface2 = selected_indices[1] + 1
    
    if interface1 > num_subnets:
        interface1 %= num_subnets
    if interface2 > num_subnets:
        interface2 %= num_subnets
    
    methods = [
        # Simple route swap
        # FIX: ip route del SUBNET dev WRONG_INTERFACE && ip route add SUBNET dev CORRECT_INTERFACE
        lambda: [router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{interface1}'),
                router.cmd(f'ip route add {subnets[selected_indices[0]][2]} dev r0-eth{interface2}')],
        
        # Add a more specific route with blackhole
        # FIX: ip route del SPECIFIC_IP/32 blackhole
        lambda: router.cmd(f'ip route add {subnets[selected_indices[0]][2].split("/")[0]}/32 blackhole'),
        
        # Add incorrect gateway
        # FIX: ip route del SUBNET via WRONG_GATEWAY && ip route add SUBNET dev CORRECT_INTERFACE
        lambda: [router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{interface1}'),
                router.cmd(f'ip route add {subnets[selected_indices[0]][2]} via 10.255.255.254')],
        
        # Add route with very high metric
        # FIX: ip route del SUBNET dev WRONG_INTERFACE metric VALUE && ip route add SUBNET dev CORRECT_INTERFACE
        lambda: [router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{interface1}'),
                router.cmd(f'ip route add {subnets[selected_indices[0]][2]} dev r0-eth{interface2} metric 10000')],
        
        # Create potential routing loop
        # FIX: ip route del SUBNET via LOOP_ADDRESS && ip route add SUBNET dev CORRECT_INTERFACE
        lambda: [router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{interface1}'),
                router.cmd(f'ip route add {subnets[selected_indices[0]][2]} via {subnets[selected_indices[1]][0].split("/")[0]}')]
    ]
    chosen_method = random.choice(methods)
    info(f'*** Injecting error: Wrong routing table from {subnets[selected_indices[0]][2]} using method {methods.index(chosen_method)+1}\n')
    chosen_method()


# Complexity control: randomly pick given number error type to inject
def inject_errors(router, subnets, error_number=1, errortype=None):
    error_functions = {
        'disable_routing': error_disable_routing,
        'disable_interface': error_disable_interface,
        'remove_ip': error_remove_ip,
        'drop_traffic_to_from_subnet': error_drop_traffic_to_from_subnet,
        'wrong_routing_table': error_wrong_routing_table
    }
    
    if errortype:
        if isinstance(errortype, list):
            errors_to_inject = [error_functions[et] for et in errortype]
        else:
            errors_to_inject = [error_functions[errortype]]
    else:
        num_errors = min(error_number, len(error_functions))
        errors_to_inject = random.sample(list(error_functions.values()), num_errors)
    
    for error in errors_to_inject:
        error(router, subnets)
    
    return errors_to_inject


# Generate detailed error information for each error type
def get_detail(error_type, hostnumber):
    # TODO: Packet loss is not stable
    if error_type == 'disable_routing':
        method = random.randint(1, 4)
        if method == 4:
            subnet_index = random.randint(1, hostnumber)
            subnet = f"192.168.{subnet_index}.0/24"
            return {"action": "Disable IP forwarding", "method": method, "subnet": subnet}
        return {"action": "Disable IP forwarding", "method": method}
        
    elif error_type == 'disable_interface':
        rand_index = random.randint(1, hostnumber)
        method = random.randint(1, 3)
        interface = f"r0-eth{rand_index}"
        detail = {"interface": interface, "method": method}
        # if method == 4:
        #     detail["loss_pct"] = random.randint(50, 95)
        return detail
        
    elif error_type == 'remove_ip':
        rand_index = random.randint(1, hostnumber)
        method = random.randint(1, 4)
        interface = f"r0-eth{rand_index}"
        detail = {"interface": interface, "method": method}
        if method == 3:
            detail["wrong_mask"] = random.choice([8, 16, 30, 31, 32])
        return detail
        
    elif error_type == 'drop_traffic_to_from_subnet':
        rand_index = random.randint(1, hostnumber)
        method = random.randint(1, 4)
        subnet = f"192.168.{rand_index}.0/24"
        detail = {"subnet": subnet, "method": method}
        if method == 4:
            detail["delay_ms"] = random.randint(1000, 5000)
        # elif method == 5:
        #     detail["corrupt_pct"] = random.randint(20, 50)
        return detail
        
    elif error_type == 'wrong_routing_table':
        # Randomly select two different interface indexes
        indexes = random.sample(range(1, hostnumber + 1), 2)
        from_subnet = f"192.168.{indexes[0]}.0/24"
        to_subnet = f"192.168.{indexes[1]}.0/24"
        del_interface = f"r0-eth{indexes[0]}"
        add_interface = f"r0-eth{indexes[1]}"
        method = random.randint(1, 4)
        
        detail = {
            "from": from_subnet, 
            "to": to_subnet, 
            "del_interface": del_interface, 
            "add_interface": add_interface,
            "method": method,
            "to_ip": f"192.168.{indexes[1]}.1"
        }
        
        if method == 4:
            detail["metric"] = random.randint(5000, 20000)
            
        return detail
        
    else:
        return {}


# Process single error with the advanced methods
def process_single_error(router, subnets, errortype, errordetail, unique_id):
    if errortype == "disable_routing":
        method = errordetail.get("method", random.randint(1, 4))
        print(f'*** Injecting error: Disabling IP forwarding using method {method}\n')
        
        if method == 1:
            print('*** Disabling IP forwarding using sysctl\n')
            router.cmd('sysctl -w net.ipv4.ip_forward=0')
        elif method == 2:
            print('*** Disabling IP forwarding using iptables\n')
            router.cmd('iptables -P FORWARD DROP')
        elif method == 3:
            print('*** Disabling IP forwarding using ip rule\n')
            router.cmd('ip rule add prohibit pref 0')
        elif method == 4:
            print('*** Disabling IP forwarding for a specific subnet\n')
            subnet_to_drop = errordetail.get("subnet", random.choice(subnets)[2])
            router.cmd(f'iptables -A FORWARD -s {subnet_to_drop} -j DROP')
        return
        
    elif errortype == "disable_interface":
        if "interface" not in errordetail:
            info("Error: not enough detailed information for disable_interface\n")
            return
            
        interface = errordetail["interface"]
        interface =f"p{unique_id % 100}_" + interface
        method = errordetail.get("method", random.randint(1, 3))
        info(f'*** Injecting error: Interfering with interface {interface} using method {method}\n')
        
        if method == 1:
            router.cmd(f'ifconfig {interface} down')
        elif method == 2:
            router.cmd(f'ip link set {interface} down')
        elif method == 3:
            router.cmd(f'ip link set {interface} mtu 68')
        return
        
    elif errortype == "remove_ip":
        if "interface" not in errordetail:
            info("Error: not enough detailed information for remove_ip\n")
            return
            
        interface = errordetail["interface"]
        
        # Extract interface index to find corresponding subnet
        try:
            interface_index = int(interface.split('eth')[1]) - 1
            print(f"Interface index: {interface_index}")
            print(f"Subnets: {len(subnets)}")
            if interface_index < len(subnets):
                print(f"Subnets: {subnets}")
                current_subnet = subnets[interface_index][0].split('/')[0]
                print(f"Current subnet: {current_subnet}")
            else:
                current_subnet = "192.168.1.1"  # Fallback
        except (IndexError, ValueError):
            current_subnet = "192.168.1.1"  # Fallback
            
        method = errordetail.get("method", random.randint(1, 4))
        info(f'*** Injecting error: Modifying IP on interface {interface} using method {method}\n')
        interface =f"p{unique_id % 100}_" + interface
        if method == 1:
            router.cmd(f'ip addr flush dev {interface}')
        elif method == 2:
            random_ip = f'10.{random.randint(1,254)}.{random.randint(1,254)}.1/24'
            router.cmd(f'ip addr flush dev {interface} && ip addr add {random_ip} dev {interface}')
        elif method == 3:
            wrong_mask = errordetail.get("wrong_mask", random.choice([8, 16, 30, 31, 32]))
            router.cmd(f'ip addr flush dev {interface} && ip addr add {current_subnet}/{wrong_mask} dev {interface}')
        elif method == 4:
            # Try to find a different subnet IP to use
            other_subnets = [s[0] for s in subnets if s[0].split("/")[0] != current_subnet]
            if other_subnets:
                duplicate_ip = random.choice(other_subnets)
            else:
                duplicate_ip = "10.0.0.1/24"  # Fallback
            router.cmd(f'ip addr flush dev {interface} && ip addr add {duplicate_ip} dev {interface}')
        return
        
    elif errortype == "drop_traffic_to_from_subnet":
        if "subnet" not in errordetail:
            info("Error: not enough detailed information for drop_traffic_to_from_subnet\n")
            return
            
        subnet_address = errordetail["subnet"]
        method = errordetail.get("method", random.randint(1, 5))
        
        # Try to find the subnet index (to determine interface)
        subnet_index = None
        for i, s in enumerate(subnets):
            if s[2] == subnet_address:
                subnet_index = i
                break
                
        interface = f"p{unique_id % 100}_" + f'r0-eth{subnet_index+1}' if subnet_index is not None else f"p{unique_id % 100}_" + f'r0-eth1'
        
        info(f'*** Injecting error: Traffic manipulation for subnet {subnet_address} using method {method}\n')
        
        if method == 1:
            router.cmd(f'iptables -A INPUT -s {subnet_address} -j DROP')
            router.cmd(f'iptables -A OUTPUT -d {subnet_address} -j DROP')
        elif method == 2:
            router.cmd(f'iptables -A INPUT -s {subnet_address} -j REJECT')
            router.cmd(f'iptables -A OUTPUT -d {subnet_address} -j REJECT')
        elif method == 3:
            router.cmd(f'iptables -A INPUT -s {subnet_address} -p icmp -j DROP')
            router.cmd(f'iptables -A OUTPUT -d {subnet_address} -p icmp -j DROP')
        elif method == 4:
            delay_ms = errordetail.get("delay_ms", random.randint(1000, 5000))
            router.cmd(f'tc qdisc add dev {interface} root netem delay {delay_ms}ms')
        # elif method == 5:
        #     corrupt_pct = errordetail.get("corrupt_pct", random.randint(20, 50))
        #     router.cmd(f'tc qdisc add dev {interface} root netem corrupt {corrupt_pct}%')
        return
        
    elif errortype == "wrong_routing_table":
        if "from" not in errordetail or "to" not in errordetail:
            info("Error: not enough detailed information for wrong_routing_table\n")
            return
            
        from_subnet = errordetail["from"]
        to_subnet = errordetail.get("to", "192.168.1.0/24")
        del_interface = errordetail.get("del_interface", "r0-eth1")
        add_interface = errordetail.get("add_interface", "r0-eth2")
        
        method = errordetail.get("method", random.randint(1, 4))
        info(f'*** Injecting error: Wrong routing table from {from_subnet} using method {method}\n')
        add_interface = f"p{unique_id % 100}_" + add_interface
        del_interface = f"p{unique_id % 100}_" + del_interface
        if method == 1:
            router.cmd(f'ip route del {from_subnet} dev {del_interface}')
            router.cmd(f'ip route add {from_subnet} dev {add_interface}')
        elif method == 2:
            # Add incorrect gateway
            router.cmd(f'ip route del {from_subnet} dev {del_interface}')
            router.cmd(f'ip route add {from_subnet} via 10.255.255.254')
        elif method == 3:
            # Add route with very high metric
            router.cmd(f'ip route del {from_subnet} dev {del_interface}')
            metric = errordetail.get("metric", 10000)
            router.cmd(f'ip route add {from_subnet} dev {add_interface} metric {metric}')
        elif method == 4:
            # Create potential routing loop
            router.cmd(f'ip route del {from_subnet} dev {del_interface}')
            
            # Extract to_subnet's IP address
            try:
                to_ip = errordetail.get("to_ip", to_subnet.split("/")[0])
            except:
                to_ip = "192.168.1.1"  # Fallback
                
            router.cmd(f'ip route add {from_subnet} via {to_ip}')
        return
        
    info(f"Error: Unknown error type: {errortype}\n")


# Generate a configuration file with a specified number of queries for each error type
def generate_config(filename='advanced_error_config.json', num_errors_per_type=20, num_switches=2, num_hosts_per_subnet=1):
    """
    Generate benchmark configuration file.
    
    Args:
        filename: Output file path
        num_errors_per_type: Number of queries per error type
        num_switches: Number of switches in the topology (default: 2)
        num_hosts_per_subnet: Number of hosts per subnet (default: 1)
    """
    queries = []
    error_types = [
        'disable_routing',
        'disable_interface',
        'remove_ip',
        'drop_traffic_to_from_subnet',
        'wrong_routing_table'
    ]
    
    # Generate queries for each error type with different methods
    for et in error_types:
        for _ in range(num_errors_per_type):
            detail = get_detail(et, num_switches)
            query = {
                "num_switches": num_switches,
                "num_hosts_per_subnet": num_hosts_per_subnet,
                "errornumber": 1,
                "errortype": et,
                "errordetail": detail
            }
            queries.append(query)
    
    # Generate combined queries (2 errors at once)
    for et1, et2 in combinations(error_types, 2):
        for _ in range(num_errors_per_type):
            detail1 = get_detail(et1, num_switches)
            detail2 = get_detail(et2, num_switches)
            query = {
                "num_switches": num_switches,
                "num_hosts_per_subnet": num_hosts_per_subnet,
                "errornumber": 2,
                "errortype": [et1, et2],
                "errordetail": [detail1, detail2]
            }
            queries.append(query)
    
    config = {"queries": queries}
    
    # Write to file
    with open(filename, 'w') as f:
        f.truncate(0)  # Clear the file content
        json.dump(config, f, indent=4)
    
    print(f"Advanced config file {filename} generated with {len(queries)} queries.")


if __name__ == "__main__":
    # Testing config generation
    generate_config() 