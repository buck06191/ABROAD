#!/bin/python
import pprint
import re
class Converter:
    """ Class to convert the SKLearn classification reports to a latex table"""
    
    def __init__(self, report):
        self.report = report
        self.encoding = {
            0: "Control",
            1: "Horizontal",
            2: "Vertical",
            3: "Pressure",
            4: "Frowning",
            5: "Ambient Light",
            6: "Torch Light"
        }
        
    
    def print_raw_report(self):
        print(repr(self.report))
        
    def strip_space(self, verbose=False):
        self.lines = []
        for l in self.report.splitlines():
            l = l.strip()
            if len(l)>0:
                self.lines.append(l)
        if verbose:
            pprint.pprint(self.lines, depth=2)           
    
    
    def conversion(self, verbose=False):
    
        def artefact_conversion(matchobj):
            i = int(matchobj.group(0))
            return self.encoding[i]
            
        self.conversion = []
        for i, l in enumerate(self.lines):
            l = re.sub('\s{2,}', ' & ', l)
            if i==0:
                l = l.title()
                l = "{0} & {1} {2}".format("Artefact",l, r"\EndTableHeader \\ \hline")
            elif i==7:
                l = "{0} {1}".format(l, r"\\ \hline")
            else:
                l = "{0}{1}".format(l,r"\\")
            l = re.sub('(?P<bold>Artefact|avg \/ total|Support)', r'\\textbf{\g<bold>}', l)
            l = re.sub(r'avg / total', r'Average/Total', l)
            l = re.sub('(^[0-6]){1}', artefact_conversion, l)   
            self.conversion.append(l)
        self.output = "\n".join(self.conversion)
        if verbose:
            print(self.output)
        return self.conversion
        


if __name__=="__main__":
    report = """             precision    recall  f1-score   support

          0       0.55      0.22      0.32       844
          1       0.04      0.01      0.02       141
          2       0.20      0.01      0.03       139
          3       0.07      0.39      0.12       140
          4       0.08      0.20      0.12       141
          5       0.93      0.80      0.86       143
          6       0.94      0.49      0.64       140

avg / total       0.47      0.27      0.31      1688"""
    conv = Converter(report)
    conv.strip_space()
    conv.conversion(verbose=True)
    
