import signal

class SigintHandlerContext(object):

    def __init__(self, message):
        self.sigint_received=False
        self.message = message

    def __enter__(self):
        signal.signal(signal.SIGINT, self._handle_sigint)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    
    def _handle_sigint(self, sig, frame):

        if self.sigint_received:
            print("\rAborted!")
            exit(1)

        self.sigint_received = True
        print(f"\r{self.message}")