import subprocess


class PIDInfo:
    """Process ps output for certain process ID (pid)"""

    def __init__(self, pid):
        ps_proc = subprocess.Popen(['ps', '-u', '-p', pid],
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE
                                   )
        out, err = ps_proc.communicate()
        self.info = out.decode().split('\n')[1]


class NvidiaSMI:
    """Process nvidia-smi output"""

    def __init__(self):
        self.gpu_names = self._gpu_names()
        self.nvidia_smi = self._nvidia_smi()
        self.general_info = self._general_info()
        self.process_info = self._process_info()

    def _nvidia_smi(self):
        """Run nvidia-smi without extra options and returns output as str"""
        proc = subprocess.Popen(['nvidia-smi'],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE
                                )
        out, err = proc.communicate()
        if err:
            raise EnvironmentError('Failed to run nvidia-smi.')
        return out.decode()

    def _gpu_names(self):
        """Run nvidia-smi to list GPUs and returns GPU names as list of str"""
        proc = subprocess.Popen(['nvidia-smi', '-L'],
                                stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE
                                )
        out, err = proc.communicate()
        if err:
            raise EnvironmentError('Failed to run nvidia-smi.')
        return out.decode().split('\n')[:-1]

    def _general_info(self):
        """Parse the general info part of the nvidia-smi output and for each
        GPU extracts operating parameters (temperature, memory use, etc.),
        returned as a list with for each GPU a dictionary"""

        def parse(string):
            if 'N/' in string:
                return -1
            else:
                return string

        header = self.nvidia_smi.split('Processes:')[0]
        lines = header.split('\n')

        # Lines 7:-3 actually contain the parameters
        gpu_info = lines[7:-3]

        general_info = []
        for i in range(len(gpu_info) // 3):
            name_line = gpu_info[3 * i].split()
            info_line = [' '] + gpu_info[3 * i + 1][1:].split()

            d = {}
            d['gpu_id'] = int(parse(name_line[1]))
            d['name'] = ' '.join(self.gpu_names[i].split()[:-2])
            d['persistence_m'] = parse(name_line[-7])
            d['bus_id'] = parse(name_line[-5])
            d['disp_a'] = parse(name_line[-4])
            d['uncorr_ecc'] = parse(name_line[-2])

            d['fan'] = info_line[1][:-1]
            d['temp'] = int(info_line[2][:-1])
            d['perf'] = info_line[3]
            d['pwr_usage'] = int(parse(info_line[4][:-1]))
            d['pwr_cap'] = int(parse(info_line[6][:-1]))
            d['mem_usage'] = int(parse(info_line[8][:-3]))
            d['mem_cap'] = int(parse(info_line[10][:-3]))
            d['gpu_util'] = int(parse(info_line[12][:-1]))
            d['comput_m'] = info_line[13]

            general_info.append(d)
        return general_info

    def _process_info(self):
        """Combine the information from nvidia-smi and the pid info for every
        pid, enabling printing the pid, username, gpu memory, ram, cpu etc."""
        process_lines = self.nvidia_smi.split('=====|')[-1].split('\n')[1:-2]
        process_list = []
        for line in process_lines:
            try:
                (_, gpu_id, pid, tp, name, gpu_mem, _) = line.split()
                ps_line = PIDInfo(pid).info
                (user, pid, cpu, mem, vsz, rss, tty, stat, start, time) = \
                    ps_line.split()[:10]
                command = ' '.join(ps_line.split()[10:])
                process_list.append({
                    'gpu_id': gpu_id,
                    'pid': pid,
                    'type': tp,
                    'name': name,
                    'gpu_mem': gpu_mem[:-3],
                    'cpu': cpu,
                    'mem': mem,
                    'user': user,
                    'command': command
                })
            except ValueError:
                pass
        return process_list


if __name__ == '__main__':
    nvidia_smi = NvidiaSMI()
    gpu_info = nvidia_smi.general_info
    print(gpu_info)
