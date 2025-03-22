import time
import wandb


def init_wandb_parser(parser):
    parser.add_argument('--wandb', action='store_true',
                        help="use wandb or not")
    parser.add_argument('--wandb_project_name', type=str, default='PL-cross',
                        help="项目名称")
    parser.add_argument('--wandb_work_name', type=str, default='ner',
                        help="当前任务名称")
    parser.add_argument('--wandb_add_time', action='store_true',
                        help="在模型路径后添加时间")
    parser.add_argument('--wandb_model_path_name', type=str, default='output_dir',
                        help="模型路径变量名")


class MyWandb:
    def __init__(self, args):
        self.args = args
        if args.wandb_add_time:
            time_str = '_' + time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
            args.__dict__[args.wandb_model_path_name] += time_str
            args.wandb_work_name += time_str
        self.model_dir = args.__dict__[args.wandb_model_path_name]
        self.wandb = wandb.init(entity='-', project=args.wandb_project_name, name=args.wandb_work_name)
        for k, v in args.__dict__.items():
            self.wandb.config[k] = v

    def finish(self):
        self.wandb.finish()

    def log(self, data:dict):
        self.wandb.log(data)


myWandb: MyWandb = None


def get_wandb(args=None):
    global myWandb
    if args is not None:
        assert args, 'wandb 在创建时必须有 args'
        myWandb = MyWandb(args)

    return myWandb


def close_wandb():
    global myWandb
    if myWandb:
        myWandb.finish()
        myWandb = None


def wandb_log(data:dict):
    global myWandb
    if myWandb:
        myWandb.log(data)


def wandb_finish():
    global myWandb
    if myWandb:
        myWandb.finish()
