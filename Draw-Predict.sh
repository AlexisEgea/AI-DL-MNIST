#!/bin/bash

# "-----------------------------------------------------------------------------"
# "|              © Execution Project Draw Predict Digit with CNN              |"
# "| Author : Alexis EGEA                                                      |"
# "-----------------------------------------------------------------------------"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	PYTHON_CMD=python3
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"* ]]; then
 	PYTHON_CMD=python
else
	echo "Unsupported OS '$OSTYPE'"
	exit 1
fi

cd src/
$PYTHON_CMD -m main
cd ..

echo
read -p "Press any key to close the terminal window"