from yrnetwork import setting
import yrnetwork.build_luc_net
import yrnetwork.setting

def main(analyzed_net):
    if analyzed_net=='luc_net':
        yrnetwork.build_luc_net.print_str(setting.BaseConfig.DATA_PATH)
    else:
        pass


if __name__ == "__main__":
    main('luc_net')