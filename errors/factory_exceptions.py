class FactoryWithoutClassifierException(AttributeError):

    def __init__(self, classifier):
        self.message = f'No factory for a {classifier} classifier'
        self.classifier = classifier
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classifier}: {self.message}'


class FitterWithoutClassifierException(AttributeError):

    def __init__(self, classifier):
        self.message = f'No fitter for a {classifier} classifier'
        self.classifier = classifier
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classifier}: {self.message}'