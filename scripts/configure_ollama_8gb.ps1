# Preview by default; use -Apply to configure Ollama for a single 8 GiB GPU.
# Applies Ollama USER environment, not FastAPI .env. Restart Ollama Desktop.
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$settings = [ordered]@{
    'OLLAMA_HOST' = '127.0.0.1:11434'
    'OLLAMA_CONTEXT_LENGTH' = '4096'
    'OLLAMA_KEEP_ALIVE' = '5m'
    'OLLAMA_MAX_LOADED_MODELS' = '1'
    'OLLAMA_NUM_PARALLEL' = '1'
    'OLLAMA_MAX_QUEUE' = '8'
    'OLLAMA_NO_CLOUD' = '1'
}
foreach ($name in $settings.Keys) {
    $desired = $settings[$name]
    $current = [Environment]::GetEnvironmentVariable($name, 'User')
    Write-Host "$name : '$current' -> '$desired'"
    if ($Apply) {
        [Environment]::SetEnvironmentVariable($name, $desired, 'User')
    }
}
if ($Apply) {
    Write-Host 'Applied. Quit and restart Ollama Desktop to load new environment.'
} else {
    Write-Host 'PREVIEW ONLY. Re-run with -Apply to set user environment.'
}
