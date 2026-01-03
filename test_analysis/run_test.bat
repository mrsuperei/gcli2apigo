@echo off
REM Run the Gemini API streaming chunk analysis test

echo ========================================
echo Gemini API Streaming Chunk Analysis
echo ========================================
echo.

REM Check if GEMINI_API_KEY is set
if "%GEMINI_API_KEY%"=="" (
    echo ERROR: GEMINI_API_KEY environment variable is not set!
    echo.
    echo Please set it first:
    echo   Windows CMD: set GEMINI_API_KEY=your_api_key_here
    echo   PowerShell: $env:GEMINI_API_KEY="your_api_key_here"
    echo.
    pause
    exit /b 1
)

echo API Key is set (first 10 chars: %GEMINI_API_KEY:~0,10%***)
echo.
echo Running test...
echo.

go run test_streaming_chunks.go

echo.
echo ========================================
echo Test completed!
echo ========================================
echo.
pause
