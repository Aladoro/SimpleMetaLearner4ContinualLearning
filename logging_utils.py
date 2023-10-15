import csv

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

import atexit
import json
import os
import os.path as osp
import time


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def setup_loggers(d=None, namespaces=None, force=False, style=None, colors=None):  # namespaces=None,
    loggers = []
    base_output_dir = d or "experiments_temp/{}".format(int(time.time()))
    namespaces = namespaces or ['train', 'test']
    styles = style or ['inline' for _ in namespaces]
    colors = colors or ['white' for _ in namespaces]
    if not force:
        assert not osp.exists(base_output_dir)
    output_weights = "{}/weights".format(base_output_dir)
    os.makedirs(output_weights)
    for namespace, style, color in zip(namespaces, styles, colors):
        logger = Logger(base_output_dir=base_output_dir,
                        namespace=namespace,
                        output_weights=output_weights,
                        style=style,
                        color=color)
        print(colorize("Logging {} data to {}".format(namespace, logger.output_file.name),
                       'green', bold=True))
        loggers.append(logger)
    return loggers

def log_dict(logger, dictionary, prefix=''):
    for k, v in dictionary.items():
        logger.log_key_val(prefix + k, v)

def save_dict_of_lists(filename, dictionary):
    with open("{}.txt".format(filename), "w") as outfile:
       writer = csv.writer(outfile)
       writer.writerow(dictionary.keys())
       writer.writerows(zip(*dictionary.values()))

class Logger():
    def __init__(self, base_output_dir, namespace, output_weights,
                 style, color):
        self.save_it = 0
        self.headers = []
        self.current_row_data = {}
        self.base_output_dir = base_output_dir
        self.namespace = namespace
        self.output_file = open(osp.join(base_output_dir, "log_{}.txt".format(namespace)), 'w')
        atexit.register(self.output_file.close)
        self.output_weights = output_weights
        self.first_row = True
        self.style = style
        self.color = color

    def save_params(self, params):
        with open(osp.join(self.base_output_dir, 'params.json'), 'w') as out:
            out.write(json.dumps(params, indent=2, separators=(',', ': ')))

    def load_params(self, dir):
        with open(osp.join(dir, "params.json"), 'r') as inp:
            data = json.loads(inp.read())
        return data

    def log_key_val(self, key, value):
        assert key not in self.current_row_data, "key already recorded {}".format(key)
        if self.first_row:
            self.headers.append(key)
        else:
            assert key in self.headers, "key not present in headers: {}".format(key)
        self.current_row_data[key] = value

    def cprint(self, print_str, *args, **kwargs):
        if self.color is not None:
            print_str = colorize(print_str, self.color)
        print(print_str, *args, **kwargs)

    def log_inline(self, ):
        vals = []
        print_str = '{} log:'.format(self.namespace)
        for key in self.headers:
            val = self.current_row_data.get(key, "")
            if type(val) == int:
                valstr = "{}".format(val)
            elif hasattr(val, "__float__"):
                valstr = "{:.5f}".format(val)
            else:
                valstr = val
            print_str += ' - {}: {}'.format(key, valstr)
            vals.append(val)
        self.cprint(print_str)
        return vals

    def log_table(self, ):
        vals = []
        self.cprint('{} log'.format(self.namespace))
        key_lens = [len(key) for key in self.headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s = %15s |"
        n_slashes = 22 + max_key_len
        self.cprint("-" * n_slashes)
        for key in self.headers:
            val = self.current_row_data.get(key, "")
            if hasattr(val, "__float__"):
                valstr = "%8.3g" % val
            else:
                valstr = val
            self.cprint(fmt % (key, valstr))
            vals.append(val)
        self.cprint("-" * n_slashes)
        return vals

    def log_iteration(self,):
        if self.style == 'table':
            vals = self.log_table()
        else:
            vals = self.log_inline()
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.headers))
                self.output_file.write("\n")
                self.first_row = False
            self.output_file.write("\t".join(map(str, vals)))
            self.output_file.write("\n")
            self.output_file.flush()
        self.current_row_data.clear()



class Metrics():
    '''Object keeping running average/latest of relevant metrics to log. '''

    def __init__(self, *args):
        self.metrics = {arg: 0 for arg in args}
        self.latest_metrics = {arg: 0 for arg in args}
        self.samples = {arg: 1e-8 for arg in args}
        self.logged_metrics = [arg for arg in args]

    def reset(self, ):
        for arg in self.metrics:
            self.metrics[arg] = 0
            self.samples[arg] = 1e-8

    def add(self, *args):
        for arg in args:
            if arg not in self.metrics:
                self.logged_metrics.append(arg)
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8

    def update(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = 0
                self.latest_metrics[arg] = 0
                self.samples[arg] = 1e-8
            self.metrics[arg] += val
            self.samples[arg] += 1

    def set(self, **kwargs):
        for arg, val in kwargs.items():
            if arg not in self.metrics:
                self.logged_metrics += arg
                self.metrics[arg] = val
                self.samples[arg] = 1
            self.metrics[arg] = val
            self.samples[arg] = 1

    def get(self, ):
        for arg, metric_agg in self.metrics.items():
            samples = self.samples[arg]
            if samples >= 1:
                self.latest_metrics[arg] = metric_agg/samples
        return self.latest_metrics


