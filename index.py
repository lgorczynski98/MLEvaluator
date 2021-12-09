import logging
from app.Report import Report
from model_selection.model_selector import ModelSelector
import pandas as pd


logging.basicConfig(level=logging.INFO)


def main():
    report = Report()

    blob2d = ModelSelector.load('blob2d.pkl')
    blob3d = ModelSelector.load('blob3d.pkl')
    blob_many_features = ModelSelector.load('blob_many_features.pkl')
    moons = ModelSelector.load('moons.pkl')
    xor = ModelSelector.load('xor.pkl')
    digits = ModelSelector.load('digits.pkl')
    imdb = ModelSelector.load('imdb.pkl')

    report.add_page('Blobs_2D', blob2d.evaluator.get_report_elements(report.app))

    report.add_page('Blobs_3D', blob3d.evaluator.get_report_elements(report.app))

    report.add_page('Blobs_Many_Features', blob_many_features.evaluator.get_report_elements(report.app))

    report.add_page('Moons', moons.evaluator.get_report_elements(report.app))

    report.add_page('XOR', xor.evaluator.get_report_elements(report.app))

    report.add_page('Digits', digits.evaluator.get_report_elements(report.app))

    report.add_page('IMDB', imdb.evaluator.get_report_elements(report.app))

    report.prepare_elements()
    report.run()

if __name__ =='__main__':
    main()