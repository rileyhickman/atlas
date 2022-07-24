#!/usr/bin/env python

import sys
import traceback


class Logger(object):
	''' Basic Logging '''
	VERBOSITY_LEVELS = {-1: '',
						 0: ['INFO', 'FATAL'],
						 1: ['INFO', 'ERROR', 'FATAL'],
						 2: ['INFO', 'WARNING', 'ERROR', 'FATAL'],
						 3: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']}
	WRITER = {'DEBUG': sys.stdout,
			  'INFO': sys.stdout,
			  'WARNING': sys.stderr,
			  'FATAL': sys.stderr}

	# colors : https://joshtronic.com/2013/09/02/how-to-use-colors-in-command-line-output/
	GREY       = '0;37'
	WHITE      = '1;37'
	YELLOW     = '1;33',
	LIGHT_RED  = '1;31',
	RED        = '0;31'
	LIGHT_BLUE = '1;34'

	COLORS = {'DEBUG': WHITE,
			  'INFO': LIGHT_BLUE,
			  'WARNING': YELLOW,
			  'ERROR': LIGHT_RED,
			  'FATAL': RED}


	def __init__(self,
				 template,
				 verbosity=0,
				 logfile=None):
		self.template = template
		self.logfile = logfile
		if isinstance(verbosity, dict):
			verbosity = verbosity['default']
		self.verbosity        = verbosity
		self.verbosity_levels = self.VERBOSITY_LEVELS[self.verbosity]


	def log(self, message, message_type):

		if message_type in self.verbosity_levels:
			color = self.COLORS[message_type]
			error_message = None
			if message_type in ['WARNING', 'ERROR', 'FATAL']:
				error_message = traceback.format_exc()
				if not 'NoneType: None' in error_message:
					self.WRITER['message_type'].write(error_message)
			uncolored_message = f'[{message_type}] {self.template} ... {message} ...\n'
			#message = f'\x1b[{color}m{uncolored_message}]\x1b[0m]'
			message = "\x1b[%sm" % (color) + uncolored_message + "\x1b[0m"
			self.WRITER[message_type].write(message)

			return error_message, uncolored_message