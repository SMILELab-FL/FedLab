# set environment variable, "token-from-codecov"
export CODECOV_TOKEN="1a85424a-8aaa-475c-bdf7-48c2925ad12b" 

# in package dir, run test
coverage run --source=fedlab setup.py test

# generate local report
coverage report

# generate local xml report, default name is coverage.xml
coverage xml


# download codecov uploader
case "$(uname -s)" in

   Darwin)
     echo 'Mac OS X'
     curl -Os https://uploader.codecov.io/latest/macos/codecov  # for macOS
	 chmod +x codecov
	 ./codecov -t ${CODECOV_TOKEN}
     ;;

   Linux)
     echo 'Linux'
     curl -Os https://uploader.codecov.io/latest/linux/codecov  # for linux
     chmod +x codecov
     ./codecov -t ${CODECOV_TOKEN}
     ;;

   CYGWIN*|MINGW32*|MSYS*|MINGW*)
     echo 'MS Windows'
     $ProgressPreference = 'SilentlyContinue'
     Invoke-WebRequest -Uri https://uploader.codecov.io/latest/windows/codecov.exe 
     -Outfile codecov.exe
     .\codecov.exe -t ${CODECOV_TOKEN}
     ;;

   # Add here more strings to compare
   # See correspondence table at the bottom of this answer

   *)
     echo 'Other OS' 

     ;;
esac


# try to delete local xml file!!! check .gitignore file!!!!
