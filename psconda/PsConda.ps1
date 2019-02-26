<#
  Copy this file your conda installation directory: <CONDA_PREFIX>\Scripts
  dot-source this file - to use the conda commands exportd by this powwershell script.
#>


<#
.SYNOPSIS
    Wrapper to onda.exe. 
    When called with 'activate' or 'dactivate' we parse the returned temporary DOS batch file
    containg SET name=alue statements and set the equivalent powershell environment variables.
#>
function Global:Invoke-CondaCmd
{
    [CmdletBinding(SupportsShouldProcess = $true)] 
    [string[]] $CondaArgs = $Args | Where-Object { @("-whatif", "-verbose") -notcontains $_ }   

    $CondaExe = $Env:CONDA_EXE
    if (!$Env:CONDA_EXE)
    {
        $CondaExe = "$PSScriptRoot\conda.exe"
        if (!(Test-Path $CondaExe -ErrorAction Ignore))
        {
            # If this file is not in the Scripts folder then check if caller defined CONDA_PREFIX
            if ($env:CONDA_PREFIX)
            {
                $CondaExe = "$env:CONDA_PREFIX\Scripts\conda.exe"
            }
        }    
    }
    if (!(Test-Path $CondaExe -ErrorAction Ignore))
    {
        Write-Host "Can't find $CondaExe" -ForegroundColor Red
        return
    }
    Write-Verbose "CONDA_EXE = $CondaExe"
    
    [string] $CondaCmd = $null
    if ($CondaArgs)
    {
        $CondaCmd = $CondaArgs[0]
        if ($CondaArgs.Count -gt 1)
        {
            $CondaEnv = $CondaArgs[1]
        }
    }
    
    $CondaCmd2 = $CondaCmd
    if (@("activate", "deactivate") -notcontains $CondaCmd)
    {
        & $CondaExe $CondaArgs
        if ($LASTEXITCODE -ne 0)
        {
            return
        }
        $CondaCmd = "reactivate"
        $CondaCmd2 = $CondaEnv = $null
    }
    
    try
    {
        $CmdFile =  & $CondaExe shell.cmd.exe $CondaCmd $CondaCmd2 $CondaEnv
        if ($CmdFile)
        {
            [array] $CmdLineS = Get-Content -Path $CmdFile
            Remove-Item $CmdFile -Force -ErrorAction Ignore -WhatIf:$false
            [int] $CONDA_SHLVL = 0
            <# Example:
            @SET "CONDA_DEFAULT_ENV=base"
            @SET "CONDA_EXE=C:\Apps\Miniconda\Scripts\conda.exe"
            @SET "CONDA_PREFIX=C:\Apps\Miniconda"
            @SET "CONDA_PROMPT_MODIFIER=(base) "
            @SET "CONDA_PYTHON_EXE=C:\Apps\Miniconda\python.exe"
            @SET "CONDA_SHLVL=1"
            @SET "PATH=C:\Apps\Miniconda;C:\Apps\Miniconda\Library\mingw-w64\bin;C:\Apps\Miniconda\Library\usr\bin;C:\Apps\Miniconda\Library\bin;C:\Apps\Miniconda\Scripts;C:\Apps\Miniconda\bin;C:\Windows\system32;..."
            @SET "PYTHONIOENCODING=1252"
            #>
            foreach ($Cmd in $CmdLineS)
            {
                if ($Cmd -match 'SET\s+"*(?<Name>.*)=(?<Value>[^"]*)')
                {
                    $Name = $Matches.Name
                    $Value = $Matches.value
                    if (![string]::IsNullOrWhiteSpace($Name))
                    {
                        Write-Verbose "SET ENV:$Name=$Value"
                        if ([string]::IsNullOrWhiteSpace($Value))
                        {
                            $Value = $null
                        }
                        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
                        if ($Name -eq "PYTHONIOENCODING")
                        {
                            & chcp $Value | Out-Null
                        }
                        if ($Name -eq "CONDA_SHLVL")
                        {
                            $CONDA_SHLVL = $Value
                        }
                    }
                }
            }
    
            if ($CONDA_SHLVL -eq 0)
            {
                # Restore the original prompt function
                if ($Function:Prompt_PS -and ($Function:Prompt -eq $Function:Prompt_Conda))
                {
                    $Function:Prompt = $Function:Prompt_PS
                }
                # Cleanup the environment
                Get-Item Env:Conda* | Set-Item -value $null -Force -ErrorAction Ignore
                Remove-Item -Path Function:Prompt_* -ErrorAction Ignore
                Remove-Item -Path Function:Invoke-Conda* -ErrorAction Ignore
                Remove-Item -Path Alias:*conda* -Force -ErrorAction Ignore
                Remove-Item -Path Alias:*activate -Force -ErrorAction Ignore
            }
            else
            {
                # Replace the original prompt function after backing it up
                if ($Function:Prompt -ne $Function:Prompt_Conda)
                {
                    New-Item -Path Function:Prompt_PS -Value $Function:Prompt -Force | Out-Null
                }
                $Function:Prompt = $Function:Prompt_Conda
            }
        }            
    }
    Catch
    {
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}
Set-Alias -Name "conda" -Value Global:Invoke-CondaCmd -Scope Global


<#
.SYNOPSIS
    Prefixes the the prompt with the virtual environment
#>
function Global:Prompt_Conda([String] $Prefix = $Env:CONDA_PROMPT_MODIFIER, $Color = $Env:CONDA_PROMPT_COLOR)
{
    if ($Prefix)
    {
        if ([String]::IsNullOrWhiteSpace($Color))
        {
            $Color = "Cyan"
        }
        Write-Host $Prefix -ForegroundColor $Color -nonewline 
    }
    else 
    {
        Write-Host "PS " -nonewline 
    }
    Write-Host $ExecutionContext.SessionState.Path.CurrentLocation -nonewline 
    "$('>' * ($nestedPromptLevel + 1)) "    
}

function Global:Invoke-CondaActivate
{
    Invoke-CondaCmd "activate", $Args
}
Set-Alias -Name "activate" -Value Global:Invoke-CondaActivate -Scope Global

function Global:Invoke-CondaDeactivate
{
    Invoke-CondaCmd "deactivate", $Args
}
Set-Alias -Name "deactivate" -Value Global:Invoke-CondaDeactivate -Scope Global



# ------------ Simple tests ------------
<# Comment for debugging ...
    Write-Host "ENV before activate" -ForegroundColor Cyan
    Get-ChildItem Env:CONDA*
    activate
    Write-Host "ENV after activate" -ForegroundColor Cyan
    Get-ChildItem Env:CONDA*
    conda info
    deactivate
    Write-Host "ENV after deactivate" -ForegroundColor Cyan
    Get-ChildItem Env:CONDA*
#>