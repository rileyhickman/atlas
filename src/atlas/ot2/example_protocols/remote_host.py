#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from atlas import Logger

# from socket import error as socket_error
# import paramiko


# class Host(object):

# 	def __init__(self, host_ip:str, username:str, key_file_path:str, key_file_password:str):
# 		self.host_ip = host_ip
# 		self.host_username = username
# 		self.key_file_path = key_file_path
# self.key_file_password = key_file_password


# class ExampleException(Exception):
# 	# TODO: fill this out
# 	pass

# class Host(object):

# 	def __init__(self, host_ip:str, username:str, key_file_path:str, key_file_password:str):
# 		self.host_ip = host_ip
# 		self.host_username = username
# 		self.key_file_path = key_file_path
# 		self.key_file_password = key_file_password
# 		self.key_file_obj = self._get_key(self.key_file_path)


# 	def _get_key(self, key_path:str) -> paramiko.PKey:
# 		#with open(key_path) as f:
# 		return paramiko.RSAKey.from_private_key_file(key_path, password=self.key_file_password)


# 	def _get_connection(self) -> Connection:
# 		connect_kwargs = {
# 			'pkey': self.key_file_obj,
# 			# 'key_filename':self.key_file_path,
# 			# 'password': self.key_file_password,
# 			# 'allow_agent':False,
# 			# 'look_for_keys':False,
# 			# 'connection_attempts':5,
# 			'banner_timeout':60,
# 		}
# 		return Connection(
# 			host=self.host_ip,
# 			user=self.host_username,
# 			port=31950,
# 			connect_kwargs=connect_kwargs,

# 		)


# 	def run_command(self, command:str):

# 		try:
# 			with self._get_connection() as connection:
# 				msg = f'Running {command} on OT-2 robot @ {self.host_ip}'
# 				Logger.log(msg, 'INFO')
# 				result = connection.run(command, warn=True, hide='stderr')
# 		except (socket_error, AuthenticationException) as exc:
# 			self._raise_authentication_err(exc)

# 		if result.failed:
# 			raise ExampleException(
# 				f'The command {command} on OT-2 robot @ {self.host_ip} failed with the error {results.stderr}'
# 			)


# 	def _raise_authentication_err(self, exc):
# 	    raise ExampleException(
# 	        "SSH: could not connect to {host} "
# 	        "(username: {user}, key: {key}): {exc}".format(
# 	            host=self.host_ip, user=self.host_username,
# 	            key=self.key_file_path, exc=exc)
# 	    )


if __name__ == "__main__":

    import paramiko
    from paramiko import AutoAddPolicy, RSAKey, SSHClient
    from paramiko.auth_handler import AuthenticationException, SSHException
    from scp import SCPClient, SCPException

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.client.WarningPolicy)

    # pkey = paramiko.RSAKey.from_private_key_file("/Users/rileyhickman/.ssh/id_rsa")

    _ = ssh_client.connect(
        hostname="192.168.0.168",  #'192.168.0.112',#'',
        username="root",
        key_filename="/Users/rileyhickman/.ssh/id_rsa",
        # port=31950,
        # look_for_keys=False,
    )

    # test connection with command
    command = "whoami"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    print("\nPrinting stdout lines...\n")
    for line in stdout.readlines():
        print(line)

    # open scp client
    scp_client = SCPClient(ssh_client.get_transport())

    filepath = "../__TEST_run.py"
    remote_path = "/data/user_storage/"

    scp_client.put(filepath, remote_path=remote_path, recursive=True)

    # execute the file
    stdin, stdout, stderr = ssh_client.exec_command(
        f"opentrons_execute {remote_path}__TEST_run.py"
    )

    print("\nPrinting stdout lines...\n")
    for line in stdout.readlines():
        print(line)
