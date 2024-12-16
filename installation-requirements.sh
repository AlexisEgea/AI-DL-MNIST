
#!/bin/bash

echo "-----------------------------------------------------------------------------"
echo "|                          Installation Requirements                        |"
echo "| Author : Alexis EGEA                                                      |"
echo "-----------------------------------------------------------------------------"

echo "OS detected $OSTYPE"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	PYTHON_CMD=python3
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"*  ]]; then
 	PYTHON_CMD=python
else
	echo "Unsupported OS '$OSTYPE'"
	exit 1
fi

echo "_____________________________________________________________________________"
echo "creation venv repository..."
$PYTHON_CMD -m venv venv
echo "...done"
echo "_____________________________________________________________________________"
echo "activation venv..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	source venv/bin/activate
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"* ]]; then
 	venv/Scripts/activate
fi

echo "...done"
echo "_____________________________________________________________________________"
echo "installation requirement.txt"
pip install -r requirement.txt
echo "Done, project ready to be executed"

echo 
read -p "Press any key to close the terminal window"
