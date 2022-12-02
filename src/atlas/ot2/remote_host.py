#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import time
import paramiko
from paramiko import AutoAddPolicy, RSAKey, SSHClient
from paramiko.auth_handler import AuthenticationException, SSHException
from scp import SCPClient, SCPException
from socket import error as socket_error

from ot2_control import Logger


class ExampleException(Exception):
  # TODO: fill this out
  pass

class Host(object):
    def __init__(self, hostname:str, username:str, key_filename:str):
        self.hostname = hostname
        self.username = username
        self.key_filename = key_filename

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(
            paramiko.client.WarningPolicy
        )
        self.get_ssh_connection()

        self.scp_client = SCPClient(self.ssh_client.get_transport())

    def get_ssh_connection(self):
        self.ssh_client.connect(
                hostname=self.hostname,
                username=self.username,
                key_filename=self.key_filename,
            )

    def run_command(self, command:str):
        msg = f'Running {command} on OT-2 robot @ {self.hostname}'
        Logger.log(msg, 'INFO')
        stdin, stdout, stderr = self.ssh_client.exec_command(command)

        Logger.log('Command stdout...', 'INFO')
        for line in stdout.readlines():
            print(line)

        # except (socket_error, AuthenticationException) as exc:
        #     #self._raise_authentication_err(exc)
        return stdin,stdout,stderr




    def _raise_authentication_err(self, exc):
        raise ExampleException(
            "SSH: could not connect to {host} "
            "(username: {user}, key: {key}): {exc}".format(
                host=self.hostname, user=self.username,
                key=self.key_filename, exc=exc
            )
        )


    def put_file(self, source_path:str, dest_path:str):
        self.scp_client.put(source_path, remote_path=dest_path, recursive=True)

    def get_file(self, source_file:str, dest_path:str):
        self.scp_client.get(source_file, local_path=dest_path, preserve_times=True)


    def remove_remote_file(self, dest_path:str, is_dir:bool=False):
        if is_dir:
            self.run_command(f'rm -r {dest_path}')
        else:
            self.run_command(f'rm {dest_path}')




if __name__ == '__main__':

    host = Host(
            hostname='192.168.0.168', 
            username='root', 
            key_filename='/Users/rileyhickman/.ssh/id_rsa',
        )

    filename = '__TEST_run.py'
    dest_path = '/data/user_storage/'
    host.put_file(filename, dest_path=dest_path)

    # host.remove_remote_file(dest_path=dest_path+filename, is_dir=False)

    # host.get_file('/data/user_storage/my_file.txt', dest_path='./')

    host.run_command(command=f'opentrons_simulate {dest_path}{filename}')



    # import paramiko
    # from paramiko import AutoAddPolicy, RSAKey, SSHClient
    # from paramiko.auth_handler import AuthenticationException, SSHException
    # from scp import SCPClient, SCPException

    # ssh_client = paramiko.SSHClient()
    # ssh_client.set_missing_host_key_policy(paramiko.client.WarningPolicy)

    # # pkey = paramiko.RSAKey.from_private_key_file("/Users/rileyhickman/.ssh/id_rsa")

    # _ = ssh_client.connect(
    #     hostname="192.168.0.168",  #'192.168.0.112',#'',
    #     username="root",
    #     key_filename="/Users/rileyhickman/.ssh/id_rsa",
    #     # port=31950,
    #     # look_for_keys=False,
    # )

    # # test connection with command
    # command = "whoami"
    # stdin, stdout, stderr = ssh_client.exec_command(command)
    # print("\nPrinting stdout lines...\n")
    # for line in stdout.readlines():
    #     print(line)

    # # open scp client
    # scp_client = SCPClient(ssh_client.get_transport())

    # filepath = "../__TEST_run.py"
    # remote_path = "/data/user_storage/"

    # scp_client.put(filepath, remote_path=remote_path, recursive=True)

    # # execute the file
    # stdin, stdout, stderr = ssh_client.exec_command(
    #     f"opentrons_execute {remote_path}__TEST_run.py"
    # )

    # print("\nPrinting stdout lines...\n")
    # for line in stdout.readlines():
    #     print(line)
