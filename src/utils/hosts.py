import requests
import os


def get_hostname():
    return os.getenv('HOSTNAME', 'localhost')


def get_public_ipv4_ubuntu():
    """
    Get the public IPv4 address of the machine using OpenDNS.
    This function uses the `dig` command to query OpenDNS for the public IP address.
    It is needed to identify the machines launced manually in tracking systems as Sentry, Cloudwatch.

    Returns:
        str: The public IPv4 address of the machine. If it is null, the inference is running on kubernetes pods.
    """
    try:
        response = requests.get('https://api4.ipify.org', timeout=3)
        if response.status_code == 200:
            return response.text.strip()
    except:
        return "IP_Not_Found"
