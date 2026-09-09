param([string]$InputDocx,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$app=$null
$doc=$null
try {
    $app=New-Object -ComObject Word.Application
    $app.Visible=$false
    $app.DisplayAlerts=0
    $doc=$app.Documents.Open($InputDocx,$false,$true)
    $doc.ExportAsFixedFormat($OutputPdf,17)
} finally {
    if($null -ne $doc){$doc.Close(0);[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)}
    if($null -ne $app){$app.Quit();[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)}
}
