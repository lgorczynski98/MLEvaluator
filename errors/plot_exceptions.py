class TooManyDimensionsToPlotException(Exception):
    ''' Exception raised when more than 2 dimension data is about to be plotted on 2 or 3 dimension graph '''

    def __init__(self):
        self.message = 'Only 2 and 3 dimension data can be plotted'
        super().__init__(self.message)