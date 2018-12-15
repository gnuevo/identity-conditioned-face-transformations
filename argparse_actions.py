import argparse
import json

class Actions:
    @staticmethod
    def ReadFromFile():
        """Generates an argparse.Action that loads configuration arguments
        from a file

        Returns: argparse.Action

        """
        # define class
        class ReadFromFileAction(argparse.Action):
            """Opens the file in the value of the argument, reads the
            'command' field and loads all the key/value pairs to the namespace
            """
            def __init__(self, option_strings, dest, nargs=None, **kwargs):
                if nargs is not None:
                    raise ValueError("nargs not allowed")
                super(ReadFromFileAction, self).__init__(option_strings, dest,
                                                 **kwargs)

            def __call__(self, parser, namespace, values, option_string=None):
                with open(values, 'r') as f:
                    config = json.load(f)
                    for key, value in config['command'].items():
                        if key is self.dest: continue
                        setattr(namespace, key, value)
                print('%r %r %r' % (namespace, values, option_string))
                setattr(namespace, self.dest, values)
        # return class
        return ReadFromFileAction