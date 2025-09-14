@echo off
echo =======================================================
echo = Windows Virtual Memory Settings Fix for YOLOv8 Training =
echo =======================================================
echo.
echo This script will help you increase Windows virtual memory settings
echo to fix "paging file too small" errors during YOLOv8 training.
echo.
echo Please follow these manual steps:
echo.
echo 1. Right-click on "This PC" or "My Computer" and select "Properties"
echo 2. Click on "Advanced system settings"
echo 3. In the System Properties window, go to the "Advanced" tab
echo 4. Click on "Settings" under the "Performance" section
echo 5. Go to the "Advanced" tab in the Performance Options window
echo 6. Click on "Change" under the "Virtual memory" section
echo 7. Uncheck "Automatically manage paging file size for all drives"
echo 8. Select the drive where Windows is installed (usually C:)
echo 9. Select "Custom size" and set:
echo    - Initial size: 16384 MB (16 GB)
echo    - Maximum size: 32768 MB (32 GB)
echo.
echo 10. Click "Set" and then "OK" to close all windows
echo 11. Restart your computer for changes to take effect
echo.
echo After restarting, your YOLOv8 training should work without memory errors.
echo.
pause 