
import humanize
import datetime as dt
import common.defaults as defaults


def create_log_function(prefix):
    p = prefix + "|"
    return lambda x: print(p, x)



# * SECTION: DEFINE LOG >

def format_delta_human(delta):
    s = humanize.naturaldelta(
        dt.datetime.now() - log.start_time, minimum_unit="seconds")

def format_delta(delta):
    s = delta.total_seconds()
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
    
    

def log(msg, force_time=False):
    if type(msg) is not str:
        msg = str(msg)
    column_size = 8
    s = format_delta(dt.datetime.now() - log.start_time)
    # s = 'starting ' if s == 'a moment' else s
    if s == log.lastmsg and not force_time:
        s = ''
        log.repetitions = log.repetitions + 1
        if log.repetitions == 10:
            log.lastmsg = ''
    else:
        log.repetitions = 1
        log.lastmsg = s
    
    if len(s) < 10:
        line = s.ljust(column_size+1) + '| ' + msg 
        print(line, flush=True)
    else:
        print(f'{s} | {msg}', flush=True)

def logsec():
    log(3 * '-  ',  force_time=True)

log.start_time = dt.datetime.now()
log.lastmsg = ''
log.repetitions = 0
# DEFINE LOG <

# * CLASS FAKE RUN (NEPTUNE.AI)
class FakeRun():
    def __init__(self, log_fn=print):
        self.called = False
        self.log_fn = log_fn

    def __getitem__(self, k):
        return self
    
    def __setitem__(self, k, v):
        self.warning()
        
    def log(self, v):
        self.warning()
    
    def warning(self):
        if not self.called:
            self.called = True
            self.log_fn('THIS IS A FAKE RUN: NO DATA IS BEING LOGGED - this message will not be printed again')

    def __str__(self):
        return 'fake remote logger'        
# < class FakeRunnner

def add_list_of_mesh(images_df):
    vals = images_df.apply(lambda x: [t for y in x["major_mesh"].split(defaults.seq_separator) for t in y.split('/')], axis=1)
    images_df["major_mesh_l"] = vals
