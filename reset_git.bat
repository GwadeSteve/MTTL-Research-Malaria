@echo off
echo ========================================
echo Git Repository Clean Reset
echo ========================================
echo.

echo Step 1: Uninstalling Git LFS...
git lfs uninstall
echo.

echo Step 2: Removing all files from Git tracking...
git rm -rf --cached .
echo.

echo Step 3: Cleaning .gitattributes...
type nul > .gitattributes
echo.

echo Step 4: Re-adding files (respecting .gitignore)...
git add .
echo.

echo Step 5: Checking status...
git status
echo.

echo ========================================
echo Done! Review the changes above.
echo If everything looks good, run:
echo   git commit -m "Clean reset: remove LFS and ignored files"
echo   git push origin exp-branch --force
echo ========================================
pause