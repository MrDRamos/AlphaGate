$SrcDir = "D:\Lobby\AlphaPilot\AlphaGate\psconda"
$FileNameS = (Get-ChildItem -Path $SrcDir  *conda*).Name
$CmdS = $FileNameS | Foreach-Object { "New-Item -ItemType SymbolicLink -Target `"$SrcDir\$_`" -name `"$_`" "}

dir $SrcDir  | Unblock-File
Write-Output "CD to destination folder. `nRun these commands as Administrator"
$CmdS
