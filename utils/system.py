import os
from multiprocessing import Process


def kill_process_by_name(name):
    cmd = "ps -e | grep %s" % name
    f = os.popen(cmd)
    txt = f.readlines()

    if len(txt) == 0:
        print("no process \"%s\"!!" % name)
        return
    else:
        for line in txt:
            colum = line.split()
            pid = colum[0]
            cmd = "kill -9 %d" % int(pid)
            rc = os.system(cmd)
            if rc == 0:
                print("exec \"%s\" success!!" % cmd)
            else:
                print("exec \"%s\" failed!!" % cmd)

    return


def run_glances(filename):
    os.system("glances -q -t 0.5 --export csv --export-csv-file {}".format(filename))


def monitor_system_and_save(filename='glances.csv'):
    p = Process(target=run_glances, args=(filename,))
    p.daemon = True
    p.start()


def plot_result(filename='glances.csv'):
    import pandas as pd
    from matplotlib.pylab import plt
    data = pd.read_csv(filename)
    mem_used = data['mem_used'].apply(lambda x: float(x)/(10**9))

    plt.figure(1)

    plt.subplot(121)
    mem_used.plot(color="r", linestyle="-", linewidth=1)
    plt.xlabel('time step')
    plt.ylabel('GB')
    plt.title('mem_used')

    plt.subplot(122)
    gpu_0_proc = data['gpu_0_proc']
    gpu_0_proc.plot(color="b", linestyle="-", linewidth=1)
    plt.xlabel('time step')
    plt.ylabel('proc')
    plt.title('gpu_0_proc')
    plt.show()

    print("mean mem_used:{},mean_gpu_0_proc:{}".format(mem_used.mean(),gpu_0_proc.mean()))


if __name__ == '__main__':
    # import time
    # monitor_system_and_save()
    # time.sleep(5)
    plot_result()
