# QSAR Project for PH421

Initial idea was to predict cardiotoxicity, but the channel protein CHEMBL240 is not retrieved properly (?) from the database. Any API call either times out or runs infinitely.
Instead, downloaded the dataset directly from the [ChEMBL page on the bioactivity data for target CHEMBL240](https://www.ebi.ac.uk/chembl/explore/activities/eyJkYXRhc2V0Ijp7ImVudGl0eUlEIjoiQWN0aXZpdHkiLCJpbml0aWFsUXVlcnkiOnsicXVlcnkiOnsiYm9vbCI6eyJtdXN0IjpbeyJxdWVyeV9zdHJpbmciOnsicXVlcnkiOiJ0YXJnZXRfY2hlbWJsX2lkOkNIRU1CTDI0MCBBTkQgc3RhbmRhcmRfdHlwZTpcIklDNTBcIiJ9fV19fX0sImZhY2V0c1N0YXRlIjpudWxsLCJjdXN0b21GaWx0ZXJpbmciOiJ0YXJnZXRfY2hlbWJsX2lkOkNIRU1CTDI0MCBBTkQgc3RhbmRhcmRfdHlwZTpcIklDNTBcIiIsInN1YnNldEh1bWFuRGVzY3JpcHRpb24iOiJCaW9hY3Rpdml0eSBkYXRhIGZvciB0YXJnZXQgQ0hFTUJMMjQwIChWb2x0YWdlLWdhdGVkIGlud2FyZGx5IHJlY3RpZnlpbmcgcG90YXNzaXVtIGNoYW5uZWwgS0NOSDIpIC0gSUM1MCIsImV4YWN0VGV4dEZpbHRlcnMiOnt9fX0%3D), stored in `chembl_herg_data.csv`.

#### References
[1]: K. Thai, G. F. Ecker, _A binary QSAR model for classification of hERG potassium channel blockers_, Bioorganic & Medicinal Chemistry, Volume 16, Issue 7, https://doi.org/10.1016/j.bmc.2008.01.017