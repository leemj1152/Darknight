$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

python main.py predict-today `
  --input data/results.csv `
  --url "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?gmId=G101" `
  --browser `
  --output-dir reports `
  --cache-dir .cache
