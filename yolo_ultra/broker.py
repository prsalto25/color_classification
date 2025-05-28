import sys
import zmq

def main(portIn, portOut):
    """ main method """

    context = zmq.Context()

    # Socket facing clients
    frontend = context.socket(zmq.ROUTER)
    frontend.bind("tcp://*:{}".format(portIn))

    # Socket facing services
    backend = context.socket(zmq.DEALER)
    backend.bind("tcp://*:{}".format(portOut))

    zmq.proxy(frontend, backend)

    # We never get here…
    frontend.close()
    backend.close()
    context.term()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

