#!/usr/bin/python
import argparse
import json
import logging
import os
import sys
import yaml

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error('Uncaught exception:', exc_info=(
            exc_type, exc_value, exc_traceback))

def remove_logs():
    for log in ['root.log', 'main.log', 'parameters.log']:
        if os.path.exists(log):
            os.remove(log)

def set_verbosity(level, logger):
    if level:
        logger.info('Setting console logging level to {}.'.format(
                level))
        for handler in logger.root.handlers:
            if handler.get_name() == 'console':
                handler.setLevel(level)
    # if level:
    #     logger.setLevel(level)
    return logger

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'basic': {
            'format': '%(name)s - %(levelname)s - %(message)s'
            },
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'NOTSET',
            'formatter': 'basic'
            },
        'root_file_handler': {
            'class': 'logging.FileHandler',
            'level': 'NOTSET',
            'formatter': 'simple',
            'filename': 'root.log'
            }
        #     },
        # 'main_file_handler': {
        #     'class': 'logging.FileHandler',
        #     'level': 'NOTSET',
        #     'formatter': 'simple',
        #     'filename': 'main.log'
        #     },
        # 'parameters_file_handler': {
        #     'class': 'logging.FileHandler',
        #     'level': 'NOTSET',
        #     'formatter': 'simple',
        #     'filename': 'parameters.log'
        #     }
        },
    'loggers': {
        '__main__': {
            'level': 'NOTSET',
            'propagate': True
            },
        'calculate': {
            'level': 'NOTSET',
            'propagate': True
            },
        'evaluate': {
            'level': 'NOTSET',
            'propagate': True
            },
        'filetypes': {
            'level': 'NOTSET',
            'propagate': True
            },
        'genetic': {
            'level': 'NOTSET',
            'propagate': True
            },
        'gradient': {
            'level': 'NOTSET',
            'propagate': True
            },
        'loop': {
            'level': 'NOTSET',
            'propagate': True
            },
        'parameters': {
            'level': 'NOTSET',
            'propagate': True
            },
        'simplex': {
            'level': 'NOTSET',
            'propagate': True
            }
        },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'root_file_handler'],
        'propagate': False
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick script to recreate the logging configuration ' +
        'file. Also holds other logging related functions.')
    parser.add_argument(
        '--json', '-j', type=str, nargs='?', const='logging.json',
        metavar='filename',
        help='Output logging configuration using JSON. Default file is ' +
        '"logging.json".')
    parser.add_argument(
        '--yaml', '-y', type=str, nargs='?', const='logging.yaml',
        metavar='filename', default='logging.yaml',
        help='Output logging configuration using YAML. Default file is ' +
        '"logging.yaml".')
    options = parser.parse_args(sys.argv[1:])
    if options.json:
        with open(options.json, 'w') as f:
            json.dump(logging_config, f, indent=4)
    if options.yaml:
        with open(options.yaml, 'w') as f:
            yaml.dump(logging_config, f)
