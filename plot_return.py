import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

timescale = 1000


parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str, default="dreamer.out")
parser.add_argument("--window", "-w", type=int, default=1)
args = parser.parse_args()

with open(args.output, 'rb') as f:
    data = pkl.load(f)
    config = data['config']
    returns = data['avg_returns']
    termination_times = data['times']

returns = np.convolve(returns,np.ones(args.window)/args.window, mode='valid')
termination_times = np.convolve(termination_times,np.ones(args.window)/args.window, mode='valid')/timescale

data_frame = pd.DataFrame({"return":returns,"time":termination_times})

plot = sns.lineplot(x="time",y="return",data=data_frame)

plt.xlabel('')
plt.ylabel('')

plt.show()
