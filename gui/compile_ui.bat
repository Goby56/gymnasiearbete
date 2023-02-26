@echo off
for %%f in (.\xml\*) do (pyside6-uic %%f -o .\gen\%%~nf%.py -g python)