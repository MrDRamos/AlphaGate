if (!$env:CONDA_PREFIX)
{
    $env:CONDA_PREFIX="C:\Apps\Miniconda"
}
. $PSScriptRoot\PsConda.ps1
conda activate
if ($Args)
{
    if ($Args.Count -eq 1)
    {
        conda activate $Args[0]
        python --version
    }
    else
    {
        python --version
        conda $Args
    }
}
else
{
     python --version
}
