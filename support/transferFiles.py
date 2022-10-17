import os
from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()

server = 'z5208494@katana.restech.unsw.edu.au'
username = 'z5208494'
password = 'Hobbitb0i369'

target_path = "/srv/scratch/z5208494"
local_path = ""

with SCPClient(ssh.get_transport()) as scp:
    scp.put('my_file.txt', 'my_file.txt')  # Copy my_file.txt to the server

ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()


out_u_file = "../data/imported_u_train_files.csv"
lines = tuple(open(out_u_file, 'r'))

for line in lines:
    file = line.split(' ')[0]
    localpath = target_path + file + ".wav"
    remotepath = local_path + file + ".wav"
    print(remotepath)
    sftp.put(localpath, remotepath)
sftp.close()
ssh.close()
print("done")
