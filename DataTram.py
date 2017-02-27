"""
.. module:: DataTram

DataTram
*************

:Description: DataTram

    Reads the information of the Traffic status from the "dadestram" files

:Authors: shreyas
    

:Version: 

:Created on: 14/11/2016 14:13 

"""

__author__ = 'shreyas'


class DataTram:

    dt = None
    date = None

    def __init__(self, fname):
        """

        """
        f = open(fname, 'r')
        self.dt = {}
        for line in f:
            tram, date, cl, pcl = line.split('#')
            self.dt[int(tram)] = (int(cl), int(pcl))
        self.date = int(date[0:-2])

        f.close()
