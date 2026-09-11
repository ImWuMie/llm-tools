$HostName = if ($env:VLLM_HOST -and $env:VLLM_HOST -ne "0.0.0.0") { $env:VLLM_HOST } else { "127.0.0.1" }
$Port = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$Key = if ($env:VLLM_API_KEY) { $env:VLLM_API_KEY } else { "sk-local" }
$Model = if ($args.Count -gt 0) { $args[0] } else { "default" }
$Body = @{
    model = $Model
    messages = @(@{ role = "user"; content = "用一句话介绍 LoRA。" })
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://${HostName}:${Port}/v1/chat/completions" -Method Post -ContentType "application/json" -Headers @{ Authorization = "Bearer $Key" } -Body $Body
