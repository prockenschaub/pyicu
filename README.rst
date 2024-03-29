.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/pyicu.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/pyicu
    .. image:: https://readthedocs.org/projects/pyicu/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://pyicu.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/pyicu/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/pyicu
    .. image:: https://img.shields.io/pypi/v/pyicu.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/pyicu/
    .. image:: https://img.shields.io/conda/vn/conda-forge/pyicu.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/pyicu
    .. image:: https://pepy.tech/badge/pyicu/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/pyicu
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/pyicu

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
pyicu
=====


    A Python implementation of an ICU data access, harmonization, and cohort creation pipeline based on https://github.com/eth-mds/ricu.


Pyicu is based on the R package ricu (https://github.com/eth-mds/ricu) and works with ICU data from the online accessible databases MIMIC-III, MIMIC-IV, eICU, HiRid and AmsterdamUMCdb as input. It can perform data harmonization and create patient cohorts for further processing (e.g., training of machine learning algorithms) based on user specifications. With extensibility in mind, datasets and concepts can be added by providing a configuration file.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
