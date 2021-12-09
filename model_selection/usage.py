class Usage(object):
    
    def __init__(self):
        self.time_durations = []
        self.memory_usages = []

    @property
    def avg_time_duration(self):
        try:
            return float(sum(self.time_durations) / len(self.time_durations))
        except ZeroDivisionError:
            return 0

    @property
    def avg_memory_usage(self):
        try:
            return float(sum(self.memory_usages) / len(self.memory_usages))
        except ZeroDivisionError:
            return 0

    def add_time(self, time):
        self.time_durations.append(time)

    def add_memory(self, memory):
        self.memory_usages.append(memory)