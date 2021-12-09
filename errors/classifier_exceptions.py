class ClassifierWithoutPropertyException(AttributeError):
    ''' Base class of exceptions when trying to access attribute that is not a property of a class '''

    def __init__(self, classifier):
        self.classifier = classifier
        super().__init__(self.message)

    def __str__(self):
        return f'{self.classifier}: {self.message}'

class ClassifierWithoutCostException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain 'cost_' attribute
    '''

    def __init__(self, classifier):
        self.message = "Classifier does not contain 'cost_' property"
        super().__init__(classifier)

class ClassifierWithoutErrorsException(Exception):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain 'errors_' attribute
    '''

    def __init__(self, classifier):
        self.message = "Classifier does not contain 'errors_' property"
        super().__init__(classifier)

class ClassifierWithoutMetricException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain applicable metric
    '''

    def __init__(self, classifier, metric):
        self.message = f"Classifier does not apply '{metric}' metric"
        super().__init__(classifier)

class ClassifierWithoutRegularizationException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain applicable regularization
    '''

    def __init__(self, classifier, regularization):
        self.message = f"Classifier does not apply '{regularization}' regularization"
        super().__init__(classifier)

class ClassifierWithoutCriterionException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain applicable criterion
    '''

    def __init__(self, classifier, criterion):
        self.message = f"Classifier does not apply '{criterion}' criterion"
        super().__init__(classifier)

class ClassifierWithoutKernelException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain applicable criterion
    '''

    def __init__(self, classifier, kernel):
        self.message = f"Classifier does not apply '{kernel}' kernel"
        super().__init__(classifier)

class ClassifierWithoutMaxFeatureMethodException(ClassifierWithoutPropertyException):
    ''' Implementation of ClassifierWithoutPropertyException
        Raised when class does not contain method of generating max number of features
    '''

    def __init__(self, classifier, max_features):
        self.message = f"Classifier does not apply '{max_features}' max_features"
        super().__init__(classifier)