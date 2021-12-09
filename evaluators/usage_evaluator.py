import utils.plots as plots

class UsageEvaluator(object):

    def __init__(self, usages):
        self.usages = usages

    def plot_time_usage(self, short_labels=False):
        return plots.get_pie_chart(labels=[str(factory.classifier) if short_labels else repr(factory.classifier) for factory in self.usages.keys()],
                                values=[usage.avg_time_duration for usage in self.usages.values()],
                                title='Time Usage Comparison'
        )

    def plot_memory_usage(self, short_labels=False):
        return plots.get_pie_chart(labels=[str(factory.classifier) if short_labels else repr(factory.classifier) for factory in self.usages.keys()],
                                values=[usage.avg_memory_usage for usage in self.usages.values()],
                                title='Memory Usage Comparison'
        )