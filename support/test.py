import os
from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()

server = 'katana.restech.unsw.edu.au'
un = 'z5208494'
password = 'Hobbitb0i369'

target_path = "/srv/scratch/z5208494"
local_path = ""# Copy my_file.txt to the server

ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
print("here")

ssh.connect(server, username=un, password=password)
sftp = ssh.open_sftp()

print("connect")
