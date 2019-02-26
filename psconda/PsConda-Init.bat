@PowerShell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -Command "& '%~dpn0.ps1'" %*

@rem PowerShell.exe -NoProfile -Command "& {Start-Process PowerShell.exe -ArgumentList '-NoExit -NoProfile -ExecutionPolicy Bypass -File ""%~dpn0.ps1""' -Verb RunAs}"
@rem Running PowerShell scripts from a batch file
@rem http://blog.danskingdom.com/tag/batch-file/
@rem https://www.howtogeek.com/204088/how-to-use-a-batch-file-to-make-powershell-scripts-easier-to-run/
