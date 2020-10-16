import time
from yrnetwork import setting
import yrnetwork.luc_network


def main(analyzed_net):
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    if analyzed_net=='luc_net':
        yrnetwork.luc_network.build_luc_net(1987,1990)
    else:
        pass
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))


if __name__ == "__main__":
    main('luc_net')